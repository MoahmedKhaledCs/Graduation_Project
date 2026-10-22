import os

train_files = set()
test_files  = set()

for cls in os.listdir(r'D:\Project\Model\train'):
    for f in os.listdir(rf'D:\Project\Model\train\{cls}'):
        train_files.add(f)

for cls in os.listdir(r'D:\Project\Model\test'):
    for f in os.listdir(rf'D:\Project\Model\test\{cls}'):
        test_files.add(f)

common = train_files & test_files
print(f"مشتركة: {len(common)} صورة")
print(f"Train: {len(train_files)}")
print(f"Test : {len(test_files)}")