import os
import random
from env_constants import data_folder, output_folder_name, \
						  lists_folder_name, train_file_name, \
						  val_file_name, test_file_name, \
						  data_dict_file_name

dir_path = os.path.dirname(os.path.realpath(__file__))

full_data_folder = os.path.join(dir_path, data_folder)
if not os.path.exists(full_data_folder):
	raise('Data folder does not exist')

full_output_folder = os.path.join(dir_path, output_folder_name)
if not os.path.exists(full_output_folder):
	os.makedirs(full_output_folder)

lists_folder_name = os.path.join(full_output_folder, lists_folder_name)
train_file_name = os.path.join(full_output_folder, train_file_name)
val_file_name = os.path.join(full_output_folder, val_file_name)
test_file_name = os.path.join(full_output_folder, test_file_name)
data_dict_file_name = os.path.join(full_output_folder, data_dict_file_name)

VAL_RATIO = 0.2

data_dict = {}
train_lists = []
val_lists = []
test_lists = []
label = -1
# Loop each category folder
for category_folder in os.listdir(full_data_folder):
	# Update label for the category
	label = label + 1

	# Loop train and test folders
	full_category_folder_path = os.path.join(full_data_folder, category_folder)
	for folder in os.listdir(full_category_folder_path):
		prev_category_num = None
		images = []

		# Loop each image
		full_folder_path = os.path.join(full_category_folder_path, folder)
		for image in os.listdir(full_folder_path):
			# image format is category_categoryNumber_num.ext
			# category format is name1_name2_..._nameX
			name = image.split('.')[0].split('_')
			category_num = name[-2]
			category = '_'.join(name[0:-2])

			# New category number
			if category_num != prev_category_num:
				# If not first iteration, then store format data
				if prev_category_num is not None:
					output_folder = os.path.join(lists_folder_name, \
												 folder, \
												 category_folder)

					# Create the new folder for the category if it does not exist
					if not os.path.exists(output_folder):
						os.makedirs(output_folder)

					# Write to output files
					group = '%s_%s' % (category, prev_category_num)
					output_file_name = '%s.txt' % group
					output_file_name = os.path.join(output_folder, output_file_name)

					if folder == 'train':
						if random.random() > VAL_RATIO:
							train_lists.append(output_file_name)
							data_type = 'train'
						else:
							val_lists.append(output_file_name)
							data_type = 'valid'
					elif folder == 'test':
						test_lists.append(output_file_name)
						data_type = 'test'

					output = [str(label), str(len(images))] + images
					output = '\n'.join(output)

					output_file = open(output_file_name, 'w')
					output_file.write(output)

					if data_type not in data_dict:
						data_dict[data_type] = {}

					if category_folder not in data_dict[data_type]:
						data_dict[data_type][category_folder] = {}

					if group not in data_dict[data_type][category_folder]:
						data_dict[data_type][category_folder][group] = {}

					data_dict[data_type][category_folder][group]['label'] = label
					data_dict[data_type][category_folder][group]['size'] = len(images)
					data_dict[data_type][category_folder][group]['images'] = images

				# Reset parameters
				prev_category_num = category_num
				images = []
			
			# Add new image
			full_image_path = os.path.join(data_folder, category_folder, folder, image)
			images.append(full_image_path)

# Write train, val, test lists to files
train_file = open(train_file_name, 'w')
train_file.write('\n'.join(train_lists))

val_file = open(val_file_name, 'w')
val_file.write('\n'.join(val_lists))

test_file = open(test_file_name, 'w')
test_file.write('\n'.join(test_lists))

data_dict_file = open(data_dict_file_name, 'w')
data_dict_file.write(str(data_dict))