import h5py
import json

def h5_to_dict(h5file):
    result = {}
    for key in h5file:
        if isinstance(h5file[key], h5py.Group):
            result[key] = h5_to_dict(h5file[key])
        elif isinstance(h5file[key], h5py.Dataset):
            data = h5file[key][()]
            try:
                result[key] = data.tolist()  # numpy -> list
            except:
                result[key] = str(data)  # fallback
    return result

with h5py.File("./scripts/proprio_stats.h5", "r") as f:
    data_dict = h5_to_dict(f)

with open("output.json", "w") as out_file:
    json.dump(data_dict, out_file, indent=2)
