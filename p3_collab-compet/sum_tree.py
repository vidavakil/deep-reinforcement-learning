import numpy as np

class SumTree:
    """A sum-tree is similar to the array representation of a binary heap, but instead 
    of the usual heap property where the value of a parent is the maximum value of its 
    two children, in the sum-tree the value of a parent node is the sum of its children. 
    Leaf nodes store non-negative values, and the internal nodes are intermediate sums, 
    with the parent node containing the sum over all leaves.
    """
    def __init__(self, size=1024):
        """Initialize a SumTree with the given size. 
        For now, only power-of-two sizes are supported.
        
        Params
        ======
            size (int): size of the data array
        """
        
        # size needs to be a power of two
        next_power_of_two = 1
        temp = size
        while True:
            temp = temp // 2
            next_power_of_two *= 2
            if temp == 0:
                break
        if next_power_of_two / 2 != size:
            print(f"Invalid size request for a Sum-Tree = {size}. Please choose a power-of-two size!")
            assert(next_power_of_two == size)
            
        self.size = size  # maximum number of leaves in the tree that hold items
        
        self.data_size = 0
        
        # Index of the left-most leaf in the tree
        # Also points to the first entry to be filled with data.
        self.data_index = size - 1  
        
        # The first size - 1 entries in the array hold the internal nodes
        # of the tree. The last size entries in the array hold the leaves
        # of the tree. The leaves hold the actual data that we store in 
        # the tree. That is, the data array is self.tree[size - 1:]
        self.tree = np.zeros(size * 2 - 1)
        
    def update(self, index, item):
        """Update the index-th entry in the data array with item.
        
        Params
        ======
            index (int): index of the element in the data array to be updated
            item (float): the new value to be stored at the given index
        """
        
        # locate the address of the requested item in the tree
        index += self.size - 1
        
        assert(index < 2 * self.size - 1)
        change = item - self.tree[index]
        self.tree[index] = item
        parent = index
        while True:
            parent = (parent - 1) // 2
            self.tree[parent] += change
            if parent == 0:
                break

    def add(self, item):
        """Add a new item to the end of the data array. If the data array
        is already full, reset to the start of the data array and override
        older items.
        
        Params
        ======
            item (float): new item to be stored in the data array
        """
        
        self.update(self.data_index - self.size + 1, item)            
        self.data_index += 1
        if self.data_index == 2 * self.size - 1:
            # go back to the start of the data array and overwrite old items
            self.data_index = self.size - 1
        self.data_size = min(self.data_size + 1, self.size)
            
    def total_sum(self):
        """Returns the total sum of the data items, stored at location 0."""
        return self.tree[0]
    
    def largest_item(self):
        """Returns the largest data item."""
        if self.data_size == 0:
            return 0.0
        else:
            return max(self.tree[self.size - 1 : self.size - 1 + self.data_size])
    
    def weighted_sample(self, k):
        """Return the k random items and their indices in the data array,
        where each item is selected with a probability equal to
        the ratio of that item over the total sum of the data items.
        """
        assert(k > 0 and len(self) >= k)
        total_sum = self.tree[0]
        segment_size = total_sum / k
        indices = []
        items = []
        segment_start = 0.0
        for i in range(k):
            segment_end = segment_start + segment_size
            sample = np.random.uniform(segment_start, segment_end, 1)
            segment_start = segment_end
            index = self.find_index(sample)
            indices.append(index - (self.size - 1))
            items.append(self.tree[index])
        return indices, items
    
    def find_index(self, sum):
        """Find the smallest index in the data array such that the sum of
        all items in the array before that index and including that index is 
        greater than or equal to the input argument 'sum'.
        
        Params
        ======
            sum: the cumulative sum criteria for this search
        """
        
        index = 0
        while True:
            left_child = (index + 1) * 2 - 1
            right_child = left_child + 1
            if left_child >= 2 * self.size - 1:
                return index
            if sum <= self.tree[left_child]:
                index = left_child
            else:
                index = right_child
                sum = sum - self.tree[left_child]
    
    def __len__(self):
        """Return the current number of items in the data array."""
        return self.data_size        