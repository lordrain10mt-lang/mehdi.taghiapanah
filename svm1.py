import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ==========================================
# بخش اول: بارگذاری و نمایش دیتاست برای مخاطبان
# ==========================================
print("--- Loading Dataset ---")
# خواندن داده‌ها از فایل CSV تولید شده
df = pd.read_csv('breast_cancer_dataset.csv')

# نمایش ابعاد کل دیتاست (تعداد سطر و ستون)
print(f"Dataset Shape: {df.shape} (569 Samples, 30 Features + 1 Target)")

# نمایش 5 سطر اول دیتاست برای رویت ساختار و مقادیر توسط مخاطبان
print("\n--- First 5 Rows of the Dataset ---")
# برای جلوگیری از طولانی شدن خروجی، فقط ۵ ویژگی اول به همراه ستون هدف نمایش داده می‌شود
columns_to_show = list(df.columns[:5]) + ['target']
print(df[columns_to_show].head())

# نمایش توزیع کلاس‌های هدف (0: Malignant/بدخیم, 1: Benign/خوش‌خیم)
print("\n--- Target Class Distribution ---")
print(df['target'].value_counts().rename(index={0: 'Malignant (0)', 1: 'Benign (1)'}))
print("-" * 40)

# جداسازی ویژگی‌ها (X) و برچسب هدف (y)
X = df.drop('target', axis=1)
y = df['target']

# ==========================================
# بخش دوم: پیش‌پردازش داده‌ها
# ==========================================
# تقسیم داده‌ها به دو مجموعه آموزش (80%) و تست (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# مقیاس‌بندی ویژگی‌ها (Feature Scaling) - بسیار حیاتی برای الگوریتم SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# بخش سوم: تعریف و آموزش مدل SVM
# ==========================================
# استفاده از کرنل RBF برای مدیریت مرزهای غیرخطی در فضای با ابعاد بالا
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# ==========================================
# بخش چهارم: ارزیابی مدل و استخراج نتایج
# ==========================================
y_pred = svm_model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# استخراج تعداد بردارهای پشتیبان (Support Vectors)
# این بخش قدرت مدل در فشرده‌سازی اطلاعات را نشان می‌دهد
support_vectors_per_class = svm_model.n_support_
print(f"\nTotal Support Vectors: {sum(support_vectors_per_class)}")
print(f"Support Vectors per class (0, 1): {support_vectors_per_class}")

# ==========================================
# بخش پنجم: تصویرسازی مرز تصمیم (کاهش ابعاد با PCA)
# ==========================================
# از آنجا که رسم نمودار در فضای 30 بعدی ممکن نیست، ابعاد را به 2 کاهش می‌دهیم
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# آموزش یک مدل SVM جدید فقط روی دو ویژگی PCA برای رسم نمودار دوبعدی
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_pca.fit(X_train_pca, y_train)

# ساخت شبکه‌ای از نقاط (Meshgrid) برای رسم پس‌زمینه مرز تصمیم
h = .02  
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# پیش‌بینی کلاس برای هر نقطه از شبکه
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# رسم نمودار
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title('SVM Decision Boundary with RBF Kernel (PCA Reduced to 2D)')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')

# اضافه کردن راهنما (Legend)
handles, labels = scatter.legend_elements()
plt.legend(handles, ['Malignant (0)', 'Benign (1)'], loc="upper right")

plt.show()
