# import docx2txt
# import os
# import time


# input_folder = r"C:\Users\abina\OneDrive\Desktop\dckap\vol 1 - proof readed"
# output_folder = "extracted_texts"
# os.makedirs(output_folder, exist_ok=True)
# combined_output = "combined.txt"


# docx_files = [f for f in os.listdir(input_folder) if f.endswith('.docx')]

# print(f"Found {len(docx_files)} .docx files")
# print("=" * 80)

# total_start_time = time.time()

# with open(combined_output, "w", encoding="utf-8") as combined_file:

#     for idx, docx_file in enumerate(docx_files, 1):

#         input_path = os.path.join(input_folder, docx_file)
#         txt_filename = docx_file.replace('.docx', '.txt')
#         output_path = os.path.join(output_folder, txt_filename)

#         try:
#             input_size = os.path.getsize(input_path)
#             input_size_kb = input_size / 1024

#             start_time = time.time()

        
#             text = docx2txt.process(input_path)

            
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 f.write(text)

        
#             combined_file.write(f"\n\n===== {docx_file} =====\n\n")
#             combined_file.write(text)
#             combined_file.write("\n" + ("-" * 80) + "\n")

#             time_taken = time.time() - start_time

#             output_size = os.path.getsize(output_path)
#             output_size_kb = output_size / 1024

#             print(f"\n[File {idx}/{len(docx_files)}] {docx_file}")
#             print(f"   Extracted successfully")
#             print(f"   Input size:  {input_size_kb:.2f} KB")
#             print(f"   Output size: {output_size_kb:.2f} KB")
#             print(f"   Time taken: {time_taken:.3f} sec")
#             print(f"  Saved as: {txt_filename}")

#         except Exception as e:
#             print(f"\n[File {idx}/{len(docx_files)}] {docx_file}")
#             print(f"   Error: {str(e)}")

# total_time = time.time() - total_start_time

# print("\n" + "=" * 80)
# print(f"All {len(docx_files)} files processed!")
# print(f"Total time: {total_time:.3f} sec")
# print(f"Average per file: {total_time/len(docx_files):.3f} sec")
# print(f"All text combined into: {combined_output}")

import docx2txt
import os
import time

# MAIN FOLDER (contains nested subfolders)
input_folder = "Proof Readed"

output_folder = "extracted_texts"
os.makedirs(output_folder, exist_ok=True)
combined_output = "combined.txt"

# Collect ALL docx files from subfolders
docx_files = []
for root, dirs, files in os.walk(input_folder):
    for f in files:
        if f.endswith(".docx"):
            docx_files.append(os.path.join(root, f))

print(f"Found {len(docx_files)} .docx files")
print("=" * 80)

total_start_time = time.time()

with open(combined_output, "w", encoding="utf-8") as combined_file:

    for idx, input_path in enumerate(docx_files, 1):

        docx_file = os.path.basename(input_path)
        txt_filename = docx_file.replace('.docx', '.txt')

        # ▼ NEW: Create SAME folder structure for output
        relative_path = os.path.relpath(os.path.dirname(input_path), input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        output_path = os.path.join(output_subfolder, txt_filename)

        try:
            input_size = os.path.getsize(input_path)
            input_size_kb = input_size / 1024

            start_time = time.time()

            text = docx2txt.process(input_path)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            combined_file.write(f"\n\n===== {docx_file} =====\n\n")
            combined_file.write(text)
            combined_file.write("\n" + ("-" * 80) + "\n")

            time_taken = time.time() - start_time

            output_size = os.path.getsize(output_path)
            output_size_kb = output_size / 1024

            print(f"\n[File {idx}/{len(docx_files)}] {docx_file}")
            print(f"   Extracted successfully")
            print(f"   Input size:  {input_size_kb:.2f} KB")
            print(f"   Output size: {output_size_kb:.2f} KB")
            print(f"   Time taken: {time_taken:.3f} sec")
            print(f"   Saved as: {output_path}")

        except Exception as e:
            print(f"\n[File {idx}/{len(docx_files)}] {docx_file}")
            print(f"   Error: {str(e)}")

total_time = time.time() - total_start_time

print("\n" + "=" * 80)
print(f"All {len(docx_files)} files processed!")
print(f"Total time: {total_time:.3f} sec")
print(f"Average per file: {total_time/len(docx_files):.3f} sec")
print(f"All text combined into: {combined_output}")
