import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Зареждане на CSV файла
df = pd.read_csv('cars.csv')

print("Оригинални данни:")
print(df.head())

# 2. Кодиране на категориалните данни
fuel_mapping = {'gasoline': 0, 'diesel': 1, 'electric': 2}
engine_mapping = {'small': 0, 'medium': 1, 'large': 2}
weight_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
car_type_mapping = {'sport': 0, 'family': 1, 'city': 2}

df['fuel'] = df['fuel'].map(fuel_mapping)
df['engine_size'] = df['engine_size'].map(engine_mapping)
df['weight'] = df['weight'].map(weight_mapping)
df['car_type'] = df['car_type'].map(car_type_mapping)

print("\nКодирани данни:")
print(df.head())

# 3. Разделяне на данните
X = df[['doors', 'fuel', 'engine_size', 'weight']]
y = df['car_type']

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# KNN МОДЕЛ
# =========================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_accuracy = knn.score(X_test, y_test)

# =========================
# MLP МОДЕЛ
# =========================
# За MLP е добре да има скалиране
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    max_iter=2000,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)
mlp_accuracy = mlp.score(X_test_scaled, y_test)

# 5. Сравнение на резултатите
print("\n=== СРАВНЕНИЕ НА МОДЕЛИТЕ ===")
print(f"KNN точност: {knn_accuracy:.2f}")
print(f"MLP точност: {mlp_accuracy:.2f}")

if knn_accuracy > mlp_accuracy:
    print("KNN се представя по-добре върху този dataset.")
elif mlp_accuracy > knn_accuracy:
    print("MLP се представя по-добре върху този dataset.")
else:
    print("И двата модела имат еднаква точност върху този dataset.")

# 6. Прогноза за нова кола
print("\nНека въведем нова кола за предсказване:")

doors = int(input("Брой врати (примерно 2, 4, 5): "))
fuel_input = input("Тип гориво (gasoline/diesel/electric): ").strip().lower()
engine_input = input("Размер на двигател (small/medium/large): ").strip().lower()
weight_input = input("Тегло на колата (light/medium/heavy): ").strip().lower()

fuel = fuel_mapping[fuel_input]
engine_size = engine_mapping[engine_input]
weight = weight_mapping[weight_input]

new_car_df = pd.DataFrame(
    [[doors, fuel, engine_size, weight]],
    columns=['doors', 'fuel', 'engine_size', 'weight']
)

# KNN прогноза
knn_prediction = knn.predict(new_car_df)[0]

# MLP прогноза
new_car_scaled = scaler.transform(new_car_df)
mlp_prediction = mlp.predict(new_car_scaled)[0]

# Обратно към текст
reverse_car_type_mapping = {value: key for key, value in car_type_mapping.items()}

print(f"\nKNN предсказание: {reverse_car_type_mapping[knn_prediction].upper()}")
print(f"MLP предсказание: {reverse_car_type_mapping[mlp_prediction].upper()}")

# =========================================================
# ВАРИАНТ 1: Разпределение на типовете автомобили
# =========================================================
car_labels = df['car_type'].map(reverse_car_type_mapping)
counts = car_labels.value_counts()

plt.figure()
plt.bar(counts.index, counts.values)
plt.title("Разпределение на типовете автомобили")
plt.xlabel("Тип автомобил")
plt.ylabel("Брой")
plt.savefig("cars_distribution.png")
plt.show()

# =========================================================
# ВАРИАНТ 3: Confusion Matrix за KNN и MLP
# =========================================================
knn_pred = knn.predict(X_test)
mlp_pred = mlp.predict(X_test_scaled)

cm_knn = confusion_matrix(y_test, knn_pred, labels=[0, 1, 2])
disp_knn = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn,
    display_labels=['sport', 'family', 'city']
)
disp_knn.plot()
plt.title("Confusion Matrix - KNN")
plt.savefig("knn_confusion_matrix.png")
plt.show()

cm_mlp = confusion_matrix(y_test, mlp_pred, labels=[0, 1, 2])
disp_mlp = ConfusionMatrixDisplay(
    confusion_matrix=cm_mlp,
    display_labels=['sport', 'family', 'city']
)
disp_mlp.plot()
plt.title("Confusion Matrix - MLP")
plt.savefig("mlp_confusion_matrix.png")
plt.show()

# =========================================================
# ВАРИАНТ 4: Точност на KNN при различни стойности на K
# =========================================================
k_values = range(1, 9)
knn_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    knn_scores.append(score)

plt.figure()
plt.plot(k_values, knn_scores, marker='o')
plt.title("Точност на KNN при различни стойности на K")
plt.xlabel("Стойност на K")
plt.ylabel("Точност")
plt.xticks(list(k_values))
plt.ylim(0, 1.1)
plt.grid(True)
plt.savefig("knn_k_values.png")
plt.show()