import os

data_folder = 'modelnet40v1'
if not os.path.exists(data_folder):
	raise('Data folder does not exist')

output_folder_name = 'data'
if not os.path.exists(output_folder_name):
	os.makedirs(output_folder_name)

lists_folder_name = 'lists'
if not os.path.exists(lists_folder_name):
	os.makedirs(lists_folder_name)

train_file_name = 'train_lists.txt'
test_file_name = 'test_lists.txt'

lists_folder_name = os.path.join(output_folder_name, lists_folder_name)
train_file_name = os.path.join(output_folder_name, train_file_name)
test_file_name = os.path.join(output_folder_name, test_file_name)

train_lists = []
test_lists = []
label = -1
# Loop each category folder
for category_folder in os.listdir(data_folder):
	if category_folder == '.DS_Store':
		continue
	# Update label for the category
	label = label + 1

	# Loop train and test folders
	full_category_folder_path = os.path.join(data_folder, category_folder)
	for folder in os.listdir(full_category_folder_path):
		if folder == '.DS_Store':
			continue
		prev_category_num = None
		images = []

		# Loop each image
		full_folder_path = os.path.join(full_category_folder_path, folder)
		for image in os.listdir(full_folder_path):
			if image == '.DS_Store':
				continue
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
					output_file_name = '%s_%s.txt' % (category, prev_category_num)
					output_file_name = os.path.join(output_folder, output_file_name)

					if folder == 'train':
						train_lists.append('%s %d' % (output_file_name, label))
					elif folder == 'test':
						test_lists.append('%s %d' % (output_file_name, label))

					output = [str(label), str(len(images))] + images
					output = '\n'.join(output)

					output_file = open(output_file_name, 'w')
					output_file.write(output)

				# Reset parameters
				prev_category_num = category_num
				images = []
			
			# Add new image
			full_image_path = os.path.join(os.getcwd(), full_folder_path, image)
			images.append(full_image_path)

# Write train and test lists to files
train_file = open(train_file_name, 'w')
train_file.write('\n'.join(train_lists))

test_file = open(test_file_name, 'w')
test_file.write('\n'.join(test_lists))