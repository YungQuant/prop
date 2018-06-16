import os
from itertools import islice

def seriously_hugo_you_better_not_fuck_this_up():
  while True:
    # with all the files
    output_dir = '../hist_outputs'
    model_output_files = os.listdir(output_dir)
    sort_key = 'Cumulative Diff'
    top_metadata_count = 5

    result_file = 'totally_not_fucked_up_results.txt'

    # only files that show a cumulative diff
    valid_output_files = []
    for model_output_file in model_output_files:
      file_path = os.path.join(output_dir, model_output_file)
      with open(file_path) as metadata:
        some_content = "".join(list(islice(metadata, 20)))
      if sort_key not in some_content:
        continue

      blobs = []
      file_path = os.path.join(output_dir, model_output_file)
      with open(file_path) as metadata:
        blob = {}
        line_number = 0
        for line in metadata:
          line_number += 1
          if line == '\n':
            blob['Line Number'] = line_number
            if sort_key in blob:
              blobs.append(blob)
            blob = {}
          else:
            k = line.split(':')[0]
            v = line.split(':')[-1]
            blob[k] = v

      # then sort by the blobs with highest diff
      sorted_blobs = sorted(blobs, key=lambda b: b[sort_key])
      top_blobs = list(reversed(sorted_blobs))[0:top_metadata_count]

      with open(result_file, 'a') as f:
        for blob in top_blobs:
          f.write("File Name: " + model_output_file + "\n")
          f.write("Location: " + str(blob['Line Number']) + "\n")
          f.write("Cumulative Diff: " + blob[sort_key] + "\n")
          f.write("\n")

seriously_hugo_you_better_not_fuck_this_up()
