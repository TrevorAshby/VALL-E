import os
test_numbers = ['0','1','2','3','4']
# move the required files from recordings_train to recordings_test
for file in os.listdir('./archive/recordings_train/'):
    if file.split('.')[0].split('_')[-1] in test_numbers:
        os.system('mv ./archive/recordings_train/{} ./archive/recordings_test/{}'.format(file, file))
