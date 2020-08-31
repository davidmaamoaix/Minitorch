import minitorch.backend.tensor as T


class Tensor:

	def __init__(self, values):
		self.shape = shape_from_list(values)
		self.values = T._make_tensor_from_list(flatten_list(values))

	def __getitem__(self, k):
		if isinstance(k, slice):
			piece = [self.values[i] for i in range(*k.indices(len(self)))]
			return Tensor(piece)
		elif isinstance(k, tuple):
			if len(k) != len(self.shape):
				raise IndexError('Tuple index is not the same shape as tensor')

			index = 0
			for i in range(len(self.shape)):
				index += mul_all(self.shape[i + 1:]) * k[i]

			return self.values[index]

		return self.values[k]

	def __setitem__(self, index, value):
		self.values[index] = value

	def __len__(self):
		return len(self.values)

	def __repr__(self):
		return f'tensor({str(list(self.values))}, shape={self.shape})'

	def __str__(self):
		return repr(self)


def shape_from_list(values):
	x = values
	shape = []
	while isinstance(x, list):
		shape.append(len(x))
		x = x[0]

	return tuple(shape)


def flatten_list(values):
	if values == []:
		return values

	if isinstance(values[0], list):
		return flatten_list(values[0]) + flatten_list(values[1:])

	return values[:1] + flatten_list(values[1:])


def mul_all(values):
	out = 1
	for i in values:
		out *= i

	return out


def tensor(values):
	'''Constructs a new tensor from the given list.'''

	return T._make_tensor_from_list(values)


def fill_tensor(length, fill_value=0):
	'''Constructs a new tensor and fill it with the given value..'''

	return T._make_and_fill_tensor(length, fill_value)

t = Tensor([[1, 2], [3, 4], [5, 6]])
print(t[0, 1])