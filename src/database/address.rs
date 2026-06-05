/// The coordinate of one payload in the tiled database.
#[derive(Copy, Clone, Debug)]
pub struct Address {
    /// Second-dimension index (InsPIRe matrix / InsPIRe² batch).
    pub matrix: usize,
    /// First-dimension index, in single-column units.
    pub column: usize,
    /// Coefficient offset of the payload within its column (`= 0` when one
    /// payload per column, as in InsPIRe²).
    pub row_offset: usize,
}

pub type PayloadAddress = Address;

impl Address {
    /// InsPIRe query block containing [`column`](Self::column).
    pub fn block_col(&self, n: usize) -> usize {
        self.column / n
    }

    /// Column within the InsPIRe `n`-wide query block.
    pub fn col_in_block(&self, n: usize) -> usize {
        self.column % n
    }
}
