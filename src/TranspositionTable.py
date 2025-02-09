import numpy as np
from collections import OrderedDict

class TranspositionTable:

    EXACT = 0   
    LOWER = 1 
    UPPER = 2  

    class Entry:

        def __init__(self, value, depth, flag):
            
            self.value = value
            self.depth = depth
            self.flag = flag

    def __init__(self, size_mb = 8):
        
        entry_size_bytes = np.dtype([('key', np.uint64),   ('value', np.int32),  ('depth', np.uint8), ('node_type', np.uint8)]).itemsize
        self.max_size = (size_mb * 1024 * 1024) // entry_size_bytes
        self.table = OrderedDict()

    def lookup(self, zobrist_hash):
        """Look up an entry in the table by zobrist_hash."""
        
        entry = self.table.get(zobrist_hash, None)
        
        if entry:
            return (entry.value, entry.depth, entry.flag)
        else:
            return None
    
    def store(self, zobrist_hash, depth, value, flag):
        """Store an entry in the table."""
    
        if len(self.table) >= self.max_size:
            self.table.popitem(last=False)

        entry = self.table.get(zobrist_hash)
        if entry is None or entry.depth <= depth:
            self.table[zobrist_hash] = self.Entry(value, depth, flag)

    def clear(self):
        """Clear the entire table."""
        self.table.clear()