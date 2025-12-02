import os, glob

files = sorted(glob.glob("images/*.tif") + glob.glob("images/*.tiff"))

for f in files[:10]:
    print(os.path.basename(f))
