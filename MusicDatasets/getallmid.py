import os
import shutil

# Set these paths appropriately
src_root = "lmd_matched"        # Folder containing nested MIDI files
dst_root = "all_midis_flat"     # Folder to save all MIDI files

os.makedirs(dst_root, exist_ok=True)

for dirpath, _, filenames in os.walk(src_root):
    for fname in filenames:
        if fname.lower().endswith('.mid'):
            src_file = os.path.join(dirpath, fname)
            dst_file = os.path.join(dst_root, fname)
            
            # Avoid overwriting: if already exists, add a suffix
            if os.path.exists(dst_file):
                base, ext = os.path.splitext(fname)
                i = 1
                while True:
                    new_name = f"{base}_{i}{ext}"
                    new_dst = os.path.join(dst_root, new_name)
                    if not os.path.exists(new_dst):
                        dst_file = new_dst
                        break
                    i += 1
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")

print("All MIDI files have been collected.")
