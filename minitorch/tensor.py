import minitorch.minitorch_c as C

def tensor(values):
	'''Constructs a new tensor from the given list.'''

	return C._make_tensor_from_list(values)

def fill_tensor(length, fill_value=0):
	'''Constructs a new tensor and fill it with the given value..'''

	return C._make_and_fill_tensor(length, fill_value)
