"""
solution.py — Query Optimization & Indexing for pySimpleDB

Implements:
  1. BetterQueryPlanner  — selection pushdown + greedy join reordering
  2. BTreeIndex          — proper B-tree with node splitting (min degree t=3)
  3. CompositeIndex      — multi-field index built on top of BTreeIndex
  4. IndexScan           — iterates only over index-matched records
  5. IndexPlan           — Plan wrapper so IndexScan fits the plan tree
  6. IndexQueryPlanner   — detects field=constant, uses index when available
  7. create_indexes()    — builds and populates all indexes from table scans
"""

from Planner import TablePlan, SelectPlan, ProjectPlan, ProductPlan
from RelationalOp import Predicate, Term, Expression, Constant
from Record import Schema, Layout, TableScan, RecordID
from Metadata import MetadataMgr
from Transaction import Transaction


# ═══════════════════════════════════════════════════════════════════════════
#  Expression Helpers
#  -----------------------------------------------------------------
#  The textbook Java SimpleDB has Expression.isFieldName() and
#  Expression.asFieldName(). This Python port doesn't expose them,
#  so we create equivalent helpers here. All access to the internal
#  exp_value is isolated in these four functions.
# ═══════════════════════════════════════════════════════════════════════════

def _is_field(expr):
    """Check whether an Expression wraps a field name (not a constant)."""
    val = getattr(expr, "exp_value", None)
    return val is not None and not isinstance(val, Constant)


def _get_field(expr):
    """Extract the field name string.  Only valid when _is_field() is True."""
    return getattr(expr, "exp_value", None)


def _is_constant(expr):
    """Check whether an Expression wraps a constant value."""
    val = getattr(expr, "exp_value", None)
    return val is not None and isinstance(val, Constant)


def _get_constant(expr):
    """Extract the constant value.  Only valid when _is_constant() is True."""
    val = getattr(expr, "exp_value", None)
    return val.const_value if val else None


# ═══════════════════════════════════════════════════════════════════════════
#  General-Purpose Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _field_to_table(field_name, table_plans):
    """
    Given an unqualified field name (e.g. 's_id') and a dict
    {table_name: plan_node}, return the table that owns that field
    by inspecting each plan's schema.
    Returns None if the field is not found (e.g. for constants).
    """
    for tname, plan in table_plans.items():
        if field_name in plan.plan_schema().field_info:
            return tname
    return None


def _get_referenced_fields(term):
    """Return the list of field names that appear in a Term (0, 1, or 2)."""
    fields = []
    if _is_field(term.lhs):
        fields.append(_get_field(term.lhs))
    if _is_field(term.rhs):
        fields.append(_get_field(term.rhs))
    return fields


def _get_referenced_tables(term, table_plans):
    """Return the set of table names referenced by this Term."""
    tables = set()
    for f in _get_referenced_fields(term):
        t = _field_to_table(f, table_plans)
        if t is not None:
            tables.add(t)
    return tables


def _make_predicate(terms):
    """Create a Predicate from a list of Term objects."""
    pred = Predicate()
    for t in terms:
        pred.terms.append(t)
    return pred


# ═══════════════════════════════════════════════════════════════════════════
#  Shared Query-Planning Logic
#  -----------------------------------------------------------------
#  These functions are used by both BetterQueryPlanner (opt mode) and
#  IndexQueryPlanner (full mode) to avoid code duplication.
# ═══════════════════════════════════════════════════════════════════════════

def _classify_terms(terms, table_plans):
    """
    Split predicate terms into two groups:
      selection_terms — {table_name: [term, …]}  single-table predicates
      join_terms      — [term, …]                two-table join conditions
    """
    selection_terms = {}     # dict: table_name → [Term, …]
    join_terms = []          # list of Term

    for term in terms:
        tables = _get_referenced_tables(term, table_plans)

        if len(tables) == 1:
            # Involves only ONE table → can be pushed down
            tbl = next(iter(tables))
            selection_terms.setdefault(tbl, []).append(term)
        elif len(tables) >= 2:
            # Involves TWO+ tables → join condition
            join_terms.append(term)
        # else: both sides are constants — skip (always true/false)

    return selection_terms, join_terms


def _apply_selection_pushdown(table_plans, selection_terms):
    """
    Wrap each table's plan with a SelectPlan for its single-table
    predicates.  This filters rows BEFORE any cross-product, which
    drastically reduces intermediate result sizes.
    """
    for tbl_name, terms in selection_terms.items():
        if tbl_name in table_plans:
            table_plans[tbl_name] = SelectPlan(
                table_plans[tbl_name],
                _make_predicate(terms)
            )


def _greedy_join_ordering(table_plans, join_terms):
    """
    Join tables in a greedy order:
      1. Start with the table that has the fewest estimated output records.
      2. At each step, pick the next table that is connected to the
         already-joined set by a join condition.
      3. Apply the join condition immediately via SelectPlan.
      4. If no connecting join term exists, fall back to cross-product.
    """
    # Sort tables by estimated output size (smallest first)
    remaining_tables = sorted(
        table_plans.keys(),
        key=lambda t: table_plans[t].recordsOutput()
    )

    if not remaining_tables:
        return None

    current_plan = table_plans[remaining_tables[0]]
    used_tables = {remaining_tables[0]}
    remaining_tables = remaining_tables[1:]
    used_join_indices = set()

    while remaining_tables:
        best_table = None
        best_idx = None

        # Try to find a remaining table connected by a join term
        for t in remaining_tables:
            for idx, term in enumerate(join_terms):
                if idx in used_join_indices:
                    continue
                tables_in_term = _get_referenced_tables(term, table_plans)
                # This term connects table 't' to our already-used set
                if t in tables_in_term and (tables_in_term & used_tables):
                    best_table = t
                    best_idx = idx
                    break
            if best_table:
                break

        if best_table:
            # Cross-product with the chosen table, then filter with the
            # join condition immediately (avoids materialising full product)
            current_plan = ProductPlan(current_plan, table_plans[best_table])
            current_plan = SelectPlan(
                current_plan,
                _make_predicate([join_terms[best_idx]])
            )
            used_tables.add(best_table)
            used_join_indices.add(best_idx)
            remaining_tables.remove(best_table)

            # Check if any OTHER join conditions are now fully satisfiable
            for idx, term in enumerate(join_terms):
                if idx in used_join_indices:
                    continue
                tables_in_term = _get_referenced_tables(term, table_plans)
                if tables_in_term <= used_tables:          # all tables present
                    current_plan = SelectPlan(
                        current_plan,
                        _make_predicate([term])
                    )
                    used_join_indices.add(idx)
        else:
            # No join condition available → plain cross-product
            t = remaining_tables.pop(0)
            current_plan = ProductPlan(current_plan, table_plans[t])
            used_tables.add(t)

    return current_plan


def _build_optimized_plan(table_plans, terms, fields):
    """
    Full pipeline:  classify → pushdown → greedy join → project.
    Called from BetterQueryPlanner and IndexQueryPlanner (full mode).
    """
    selection_terms, join_terms = _classify_terms(terms, table_plans)
    _apply_selection_pushdown(table_plans, selection_terms)
    plan = _greedy_join_ordering(table_plans, join_terms)
    return ProjectPlan(plan, *fields)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  BetterQueryPlanner
# ═══════════════════════════════════════════════════════════════════════════

class BetterQueryPlanner:
    """
    Optimised query planner that:
      • pushes single-table selections down to individual table scans
      • reorders joins greedily (smallest table first)
    """

    def __init__(self, mm):
        self.mm = mm

    def createPlan(self, tx, query_data):
        # Step 1 — build a TablePlan for every table in FROM
        table_plans = {}
        for table_name in query_data['tables']:
            table_plans[table_name] = TablePlan(tx, table_name, self.mm)

        # Steps 2-5 — pushdown + join ordering + projection
        return _build_optimized_plan(
            table_plans,
            query_data['predicate'].terms,
            query_data['fields']
        )


# ═══════════════════════════════════════════════════════════════════════════
#  2.  BTreeIndex  —  Proper B-Tree Data Structure
# ═══════════════════════════════════════════════════════════════════════════

class BTreeNode:
    """
    A single node in the B-tree.

    For a tree of minimum degree t:
      • every node holds between  t-1  and  2t-1  keys  (root may have fewer)
      • every internal node has  len(keys)+1  children
      • all leaves sit at the same depth

    Each key position also stores a *list* of RecordIDs so that
    duplicate key values are handled naturally.
    """

    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []        # sorted key values
        self.values = []      # values[i] = [RecordID, …] for keys[i]
        self.children = []    # child pointers (only if not leaf)


class BTreeIndex:
    """
    B-tree index on a single field.

    Minimum degree t = 3  ⟹  each node holds 2 … 5 keys.

    insert(key, rid)  — O(log n), uses proactive (top-down) splitting
    search(key)       — O(log n), returns list of matching RecordIDs
    """

    MIN_DEGREE = 3          # t

    def __init__(self, tx, index_name, key_type, key_length):
        self.tx = tx
        self.index_name = index_name
        self.key_type = key_type
        self.key_length = key_length
        self.root = BTreeNode(leaf=True)

    # ── Search ───────────────────────────────────────────────────────
    def search(self, key_value):
        """Return list of RecordIDs matching key_value (empty if not found)."""
        return self._search_node(self.root, key_value)

    def _search_node(self, node, key_value):
        """Recursively descend from *node* looking for *key_value*."""
        # Binary-style scan to find position
        i = 0
        while i < len(node.keys) and key_value > node.keys[i]:
            i += 1

        # Key found at this node
        if i < len(node.keys) and key_value == node.keys[i]:
            return list(node.values[i])        # return a copy

        # Reached leaf without finding — key absent
        if node.leaf:
            return []

        # Recurse into appropriate child subtree
        return self._search_node(node.children[i], key_value)

    # ── Insert ───────────────────────────────────────────────────────
    def insert(self, key_value, record_id):
        """Insert (key_value, record_id) into the B-tree."""

        # Key is new — standard B-tree insert
        root = self.root
        if len(root.keys) == 2 * self.MIN_DEGREE - 1:
            # Root is full → grow the tree upward by one level
            new_root = BTreeNode(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_nonfull(self.root, key_value, record_id)

    def _split_child(self, parent, child_index):
        """
        Split a full child into two nodes; push the median key up
        into *parent*.

        Before:  parent.children[child_index] has  2t-1  keys  (full)
        After:   left child has  t-1  keys
                 right (new) child has  t-1  keys
                 median key moved into parent
        """
        t = self.MIN_DEGREE
        child = parent.children[child_index]
        mid = t - 1                      # index of the median key

        # New sibling gets the right half of the keys
        sibling = BTreeNode(leaf=child.leaf)
        sibling.keys = child.keys[mid + 1:]
        sibling.values = child.values[mid + 1:]
        if not child.leaf:
            sibling.children = child.children[mid + 1:]

        # Median key + its RecordID list move up into parent
        parent.keys.insert(child_index, child.keys[mid])
        parent.values.insert(child_index, child.values[mid])
        parent.children.insert(child_index + 1, sibling)

        # Truncate the original child to keep only the left half
        child.keys = child.keys[:mid]
        child.values = child.values[:mid]
        if not child.leaf:
            child.children = child.children[:mid + 1]

    def _insert_nonfull(self, node, key_value, record_id):
        """Insert into a node that is guaranteed not to be full."""
        if node.leaf:
            # Shift keys right to maintain sorted order, then insert
            # OR append if duplicate key matches!
            i = len(node.keys) - 1
            while i >= 0 and key_value < node.keys[i]:
                i -= 1

            if i >= 0 and key_value == node.keys[i]:
                # Found exact duplicate key in leaf -> append to record list
                node.values[i].append(record_id)
                return

            # Shift larger keys to the right to make space
            node.keys.insert(i + 1, key_value)
            node.values.insert(i + 1, [record_id])
        else:
            # Find child to recurse into
            i = len(node.keys) - 1
            while i >= 0 and key_value < node.keys[i]:
                i -= 1

            # Check if key is exactly matched at an internal node
            if i >= 0 and key_value == node.keys[i]:
                node.values[i].append(record_id)
                return

            i += 1  # move to correct child index

            # Proactive split: if that child is full, split it first
            if len(node.children[i].keys) == 2 * self.MIN_DEGREE - 1:
                self._split_child(node, i)
                # After split the median moved up — decide which side
                if key_value == node.keys[i]:
                    node.values[i].append(record_id)
                    return
                if key_value > node.keys[i]:
                    i += 1

            self._insert_nonfull(node.children[i], key_value, record_id)

    def close(self):
        """Release resources (no-op for in-memory tree)."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  3.  CompositeIndex  —  Multi-Field Index
# ═══════════════════════════════════════════════════════════════════════════

class CompositeIndex:
    """
    Index on two or more fields (e.g. sec_semester + sec_year).

    Built ON TOP of BTreeIndex by using Python *tuples* as keys.
    Tuple comparison is lexicographic, so the B-tree's ordering
    works correctly for composite keys of the same types.

    Example key: ('Fall', 2024)
    """

    def __init__(self, tx, index_name, field_names, field_types, field_lengths):
        self.field_names = field_names
        # Delegate storage to a standard BTreeIndex with tuple keys
        self._btree = BTreeIndex(tx, index_name, 'composite', 0)

    def insert(self, field_values, record_id):
        """field_values is a tuple like ('Fall', 2024)."""
        self._btree.insert(tuple(field_values), record_id)

    def search(self, field_values):
        """Lookup by composite key."""
        return self._btree.search(tuple(field_values))

    def close(self):
        self._btree.close()


# ═══════════════════════════════════════════════════════════════════════════
#  4.  IndexScan  —  Scan Using Index Instead of Full Table Scan
# ═══════════════════════════════════════════════════════════════════════════

class IndexScan:
    """
    Instead of reading every record in a table (O(n)),
    IndexScan uses the index to jump directly to matching records.

    Flow:
      1. At construction, call index.search(key) → list of RecordIDs
      2. nextRecord() walks through those RecordIDs, positioning the
         underlying TableScan at each one via moveToRecordID().
    """

    def __init__(self, table_scan, index, search_key):
        self.table_scan = table_scan
        self.index = index
        self.search_key = search_key
        self.matching_rids = index.search(search_key)
        self.current_pos = -1

    def beforeFirst(self):
        """Reset to before the first matching record."""
        self.current_pos = -1

    def nextRecord(self):
        """Advance to the next matching record; return False when done."""
        self.current_pos += 1
        if self.current_pos < len(self.matching_rids):
            rid = self.matching_rids[self.current_pos]
            self.table_scan.moveToRecordID(rid)
            return True
        return False

    # ── Delegate field access to the underlying TableScan ────────────
    def getInt(self, field_name):
        return self.table_scan.getInt(field_name)

    def getString(self, field_name):
        return self.table_scan.getString(field_name)

    def getVal(self, field_name):
        return self.table_scan.getVal(field_name)

    def hasField(self, field_name):
        return self.table_scan.hasField(field_name)

    def closeRecordPage(self):
        self.table_scan.closeRecordPage()


# ═══════════════════════════════════════════════════════════════════════════
#  5.  IndexPlan  —  Plan Wrapper For IndexScan
# ═══════════════════════════════════════════════════════════════════════════

class IndexPlan:
    """
    A Plan node that opens an IndexScan instead of a regular TableScan.
    Exposes the same interface (open, plan_schema, recordsOutput, …)
    so it plugs into the plan tree seamlessly.
    """

    def __init__(self, tx, table_name, mm, index, search_key):
        self.tx = tx
        self.table_name = table_name
        self.mm = mm
        self.index = index
        self.search_key = search_key
        self.layout = mm.getLayout(tx, table_name)

    def open(self):
        ts = TableScan(self.tx, self.table_name, self.layout)
        return IndexScan(ts, self.index, self.search_key)

    def blocksAccessed(self):
        return len(self.index.search(self.search_key))

    def recordsOutput(self):
        return len(self.index.search(self.search_key))

    def distinctValues(self, field_name):
        return 1          # we are searching for exactly one key value

    def plan_schema(self):
        return self.layout.schema


# ═══════════════════════════════════════════════════════════════════════════
#  6.  IndexQueryPlanner
# ═══════════════════════════════════════════════════════════════════════════

class IndexQueryPlanner:
    """
    Planner that exploits indexes for equality conditions (field = constant).

    Two modes (controlled by benchmark.py):
      • index mode  — better_planner is None → keep original join order
      • full  mode  — better_planner is set  → use pushdown + join reorder
    """

    def __init__(self, mm, indexes, better_planner=None):
        self.mm = mm
        self.indexes = indexes                # {table: {field_key: IndexObj}}
        self.better_planner = better_planner  # BetterQueryPlanner or None

    def createPlan(self, tx, query_data):
        all_terms = query_data['predicate'].terms

        # ── Step 1: build table plans, replacing with IndexPlan where we can ─
        table_plans = {}
        for table_name in query_data['tables']:
            table_plans[table_name] = TablePlan(tx, table_name, self.mm)

        consumed = set()                      # indices of terms used by index
        for table_name in query_data['tables']:
            if table_name not in self.indexes:
                continue
            table_indexes = self.indexes[table_name]

            # --- try composite indexes first (more selective) ----------------
            if self._try_composite_index(tx, table_name, table_plans,
                                         table_indexes, all_terms, consumed):
                continue                       # composite index matched

            # --- try single-field indexes ------------------------------------
            self._try_single_index(tx, table_name, table_plans,
                                   table_indexes, all_terms, consumed)

        # ── Step 2: combine tables into a single plan ────────────────────
        remaining = [t for i, t in enumerate(all_terms) if i not in consumed]

        if self.better_planner:
            # full mode: use shared optimisation pipeline
            return _build_optimized_plan(table_plans, remaining,
                                         query_data['fields'])
        else:
            # index mode: original join order, no reordering
            return self._plan_original_order(table_plans, remaining, query_data)

    # ── helper: original join order (index mode) ─────────────────────────
    def _plan_original_order(self, table_plans, remaining_terms, query_data):
        tables = query_data['tables']
        current = table_plans[tables[0]]
        for i in range(1, len(tables)):
            current = ProductPlan(current, table_plans[tables[i]])

        if remaining_terms:
            current = SelectPlan(current, _make_predicate(remaining_terms))

        return ProjectPlan(current, *query_data['fields'])

    # ── helper: try to match a composite index for this table ────────────
    def _try_composite_index(self, tx, table_name, table_plans,
                              table_indexes, all_terms, consumed):
        for field_key, idx_obj in table_indexes.items():
            if not isinstance(field_key, tuple):
                continue                       # skip single-field indexes

            # We need a 'field = constant' term for EVERY field in the key
            vals = {}
            term_indices = []
            
            # For each field in the composite key, find the first unclaimed term that matches
            for fname in field_key:
                for i, term in enumerate(all_terms):
                    if i in consumed or i in term_indices:
                        continue
                    if self._matches_field_eq_const(term, fname, table_plans[table_name]):
                        vals[fname] = self._constant_of(term)
                        term_indices.append(i)
                        break  # Found a matching term for this field, move to next field

            if len(vals) == len(field_key):
                search_key = tuple(vals[f] for f in field_key)
                table_plans[table_name] = IndexPlan(
                    tx, table_name, self.mm, idx_obj, search_key)
                consumed.update(term_indices)
                return True
        return False

    # ── helper: try to match a single-field index ────────────────────────
    def _try_single_index(self, tx, table_name, table_plans,
                           table_indexes, all_terms, consumed):
        for i, term in enumerate(all_terms):
            if i in consumed:
                continue

            fname, cval = self._extract_field_eq_const(
                term, table_plans[table_name])
            if fname is not None and fname in table_indexes:
                table_plans[table_name] = IndexPlan(
                    tx, table_name, self.mm,
                    table_indexes[fname], cval)
                consumed.add(i)
                return True                    # one index per table suffices
        return False

    # ── predicate inspection helpers ─────────────────────────────────────
    @staticmethod
    def _matches_field_eq_const(term, field_name, table_plan):
        """Does *term* say  field_name = <constant>  where field ∈ table?"""
        if field_name not in table_plan.plan_schema().field_info:
            return False
        if (_is_field(term.lhs) and _get_field(term.lhs) == field_name
                and _is_constant(term.rhs)):
            return True
        if (_is_field(term.rhs) and _get_field(term.rhs) == field_name
                and _is_constant(term.lhs)):
            return True
        return False

    @staticmethod
    def _constant_of(term):
        """Return the constant side of a field=constant term."""
        if _is_constant(term.rhs):
            return _get_constant(term.rhs)
        return _get_constant(term.lhs)

    @staticmethod
    def _extract_field_eq_const(term, table_plan):
        """
        If term is  field = constant  and field belongs to table_plan,
        return (field_name, constant_value).  Otherwise (None, None).
        """
        if _is_field(term.lhs) and _is_constant(term.rhs):
            fname = _get_field(term.lhs)
            if fname in table_plan.plan_schema().field_info:
                return fname, _get_constant(term.rhs)

        if _is_field(term.rhs) and _is_constant(term.lhs):
            fname = _get_field(term.rhs)
            if fname in table_plan.plan_schema().field_info:
                return fname, _get_constant(term.lhs)

        return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  7.  create_indexes() creating indexes from table scans
# ═══════════════════════════════════════════════════════════════════════════

def create_indexes(db, tx, index_defs=None, composite_index_defs=None):
    """
    Build and populate all indexes by scanning each table ONCE.

    Parameters
    ----------
    db                  : database object (has db.mm for metadata)
    tx                  : Transaction for reading table data
    index_defs          : {table: [(field, type, length), …]}
    composite_index_defs: {table: [((fields,…), (types,…), (lengths,…)), …]}

    Returns
    -------
    dict  {table_name: {field_key: IndexObject}}
          field_key is a str for single-field, tuple for composite
    """
    if index_defs is None:
        index_defs = {}
    if composite_index_defs is None:
        composite_index_defs = {}

    indexes = {}
    all_tables = set(list(index_defs.keys()) + list(composite_index_defs.keys()))

    for table_name in all_tables:
        indexes[table_name] = {}

        # ── Create single-field BTreeIndex objects ──
        single = []
        if table_name in index_defs:
            for (fname, ftype, flen) in index_defs[table_name]:
                idx = BTreeIndex(tx, f"idx_{table_name}_{fname}", ftype, flen)
                indexes[table_name][fname] = idx
                single.append((fname, idx))

        # ── Create CompositeIndex objects ──
        comp = []
        if table_name in composite_index_defs:
            for (fnames, ftypes, flens) in composite_index_defs[table_name]:
                idx = CompositeIndex(
                    tx, f"idx_{table_name}_{'_'.join(fnames)}",
                    fnames, ftypes, flens)
                indexes[table_name][fnames] = idx   # key is a tuple
                comp.append((fnames, idx))

        # ── Populate all indexes for this table in a single scan ──
        layout = db.mm.getLayout(tx, table_name)
        ts = TableScan(tx, table_name, layout)
        while ts.nextRecord():
            rid = ts.currentRecordID()

            for (fname, idx) in single:
                idx.insert(ts.getVal(fname), rid)

            for (fnames, idx) in comp:
                vals = tuple(ts.getVal(f) for f in fnames)
                idx.insert(vals, rid)

        ts.closeRecordPage()

    return indexes

