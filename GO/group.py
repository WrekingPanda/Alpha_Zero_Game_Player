class Group(object):

    # Initialize group 
    def __init__(self, stones=None, color=None):
        if stones is not None:
            self.stones = set(stones)
        else: 
            self.stones = set()
        
        self.border = set()
        self.color = color

    #  Add two groups of the same color
    def __add__(self, other):
        if self.color != other.color:
            raise ValueError('Can only add groups of the same color!')
        
        grp = Group(stones=self.stones.union(other.stones))
        grp.color = self.color
        grp.border = self.border.union(other.border).difference(grp.stones)
        
        return grp
    
    @property
    # returns size of the group 
    def size(self):
        return len(self.stones)
    
