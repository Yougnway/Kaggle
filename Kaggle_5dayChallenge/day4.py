# modules we'll use
import pandas as pd
import numpy as np
import fuzzywuzzy

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)

# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
print(type(before))

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")

# check the type
print(type(after))
print(after)
print(after.decode("utf-8"))
# print(after.decode("ascii"))


# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been
# replaced with the underlying byte string for the unknown character :(

before = "$, #, 你好 and नमस्ते"
after = before.encode("ascii", errors="replace")
print("ascii decode:",after.decode("ascii"))
after = before.encode("utf-8", errors="replace")
print("utf-8 decode:",after.decode("utf-8"))