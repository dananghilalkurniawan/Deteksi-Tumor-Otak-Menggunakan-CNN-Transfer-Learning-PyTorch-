# Split Dataset ke Train dan Val
base_dir = '/content/brain_tumor_dataset'

train_dir = '/content/brain_tumor_dataset_train'
val_dir = '/content/brain_tumor_dataset_val'

for folder in [train_dir, val_dir]:
    os.makedirs(os.path.join(folder, 'no'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'yes'), exist_ok=True)

def split_data(class_name):
    class_dir = os.path.join(base_dir, class_name)
    images = os.listdir(class_dir)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for img in train_imgs:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy(src, dst)

    for img in val_imgs:
        src = os.path.join(class_dir, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.copy(src, dst)

split_data('no')
split_data('yes')

print("Dataset berhasil dipisah menjadi train dan val.")
