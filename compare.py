import numpy as np
import math




bmrt_filename = "./output_ref_data.dat.bmrt"
tpurt_filename = "./output.tpuRt"

bmrt_array = np.fromfile(bmrt_filename, dtype=np.float32)
tpurt_array = np.fromfile(tpurt_filename, dtype=np.float32)

diff = abs(bmrt_array - tpurt_array)

sum = diff.sum()

print(sum)
print(sum/len(bmrt_array))