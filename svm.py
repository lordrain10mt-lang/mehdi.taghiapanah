# ==========================================
# 1. وارد کردن کتابخانه‌های مورد نیاز
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# تنظیم استایل نمودارها برای ارائه حرفه‌ای‌تر
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 2. بارگذاری و آماده‌سازی داده‌ها (Dataset)
# ==========================================
# بارگذاری دیتاست سرطان سینه (شامل 569 نمونه و 30 ویژگی عددی)
cancer = datasets.load_breast_cancer()
X = cancer.data   # ماتریس ویژگی‌ها (ابعاد: 569 در 30)
y = cancer.target # برچسب‌ها (0: بدخیم Malignant، 1: خوش‌خیم Benign)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# تقسیم داده‌ها به بخش آموزش (80%) و تست (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. مقیاس‌بندی ویژگی‌ها (Feature Scaling)
# ==========================================
# به دلیل اینکه SVM بر اساس فاصله هندسی ($||w||$) بهینه‌سازی را انجام می‌دهد،
# مقیاس‌بندی داده‌ها با میانگین 0 و واریانس 1 الزامی است.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # جلوگیری از Data Leakage

# ==========================================
# 4. تعریف و آموزش مدل SVM
# ==========================================
# استفاده از هسته RBF برای مدیریت الگوهای غیرخطی
# پارامتر C=1.0 برای کنترل جریمه خطای حاشیه نرم (Soft Margin)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# آموزش مدل با استفاده از الگوریتم بهینه‌سازی SMO
svm_model.fit(X_train_scaled, y_train)

# ==========================================
# 5. ارزیابی عملکرد مدل
# ==========================================
y_pred = svm_model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Number of Support Vectors (Class 0, Class 1): {svm_model.n_support_}")

# ==========================================
# 6. تصویرسازی مرز تصمیم (مخصوص ارائه کلاسی)
# ==========================================
# چون داده‌ها 30 بعدی هستند، برای رسم نمودار دوبعدی از PCA استفاده می‌کنیم.
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# آموزش یک مدل جدید صرفاً روی 2 ویژگی استخراج شده برای رسم نمودار
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_pca.fit(X_train_pca, y_train)

# ساخت یک شبکه مختصاتی (Meshgrid) برای رسم مرز تصمیم
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# پیش‌بینی مقادیر برای تمام نقاط شبکه
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# رسم نمودار
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                      cmap=plt.cm.coolwarm, edgecolors='k', s=40)

# متمایز کردن بردارهای پشتیبان در نمودار
support_vectors = svm_pca.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=150, 
            linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

plt.title('SVM Decision Boundary with RBF Kernel (PCA Reduced 2D Space)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.tight_layout()
plt.show()
