

import os
import time
import docx2txt
from config import INPUT_FOLDER, OUTPUT_FOLDER, COMBINED_OUTPUT_FILE

docx_files = []
for root, dirs, files in os.walk(INPUT_FOLDER):
    for f in files:
        if f.endswith(".docx"):
            docx_files.append(os.path.join(root, f))

print(f"Found {len(docx_files)} .docx files")
print("=" * 80)

total_start_time = time.time()

with open(COMBINED_OUTPUT_FILE, "w", encoding="utf-8") as combined_file:
    for idx, input_path in enumerate(docx_files, 1):

        docx_file = os.path.basename(input_path)
        txt_filename = docx_file.replace('.docx', '.txt')

        relative_path = os.path.relpath(os.path.dirname(input_path), INPUT_FOLDER)
        output_subfolder = os.path.join(OUTPUT_FOLDER, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        output_path = os.path.join(output_subfolder, txt_filename)

        try:
            input_size_kb = os.path.getsize(input_path) / 1024
            start_time = time.time()

            # Extract text
            text = docx2txt.process(input_path)

            # Save extracted text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Append to combined.txt
            combined_file.write(f"\n\n===== {docx_file} =====\n\n")
            combined_file.write(text)
            combined_file.write("\n" + ("-" * 80) + "\n")

            time_taken = time.time() - start_time
            output_size_kb = os.path.getsize(output_path) / 1024

            print(f"\n[File {idx}/{len(docx_files)}] {docx_file}")
            print("   Extracted successfully")
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
print(f"All text combined into: {COMBINED_OUTPUT_FILE}")
