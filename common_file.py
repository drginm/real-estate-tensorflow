import json

#File Tools
def generate_json_file(array, root_folder, json_filename):
    json_string = json.dumps(array)
    
    file = open(root_folder + json_filename + '.json', 'w')
    file.write(json_string)
    file.close()
