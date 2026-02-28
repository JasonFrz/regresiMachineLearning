import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    """
    Membaca data dari Excel dan membersihkan format angka (koma ke titik).
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' tidak ditemukan.")
        return None, None

    try:
        df = pd.read_excel(filename)
    except ImportError:
        df = pd.read_excel(filename)
    except Exception as e:
        print(f"Gagal membaca file: {e}")
        return None, None

    col_x_name = df.columns[0] 
    col_y_name = df.columns[1] 

    print(f"Menggunakan kolom: X={col_x_name}, Y={col_y_name}")

    if df[col_x_name].dtype == object:
        df[col_x_name] = df[col_x_name].astype(str).str.replace(',', '.').astype(float)
    
    if df[col_y_name].dtype == object:
        df[col_y_name] = df[col_y_name].astype(str).str.replace(',', '.').astype(float)

    X = df[col_x_name].values
    y = df[col_y_name].values

    return X, y

def compute_cost(X, y, theta0, theta1):
    """
    Menghitung Mean Squared Error (MSE).
    J(theta) = (1/2m) * sum((h(x) - y)^2)
    """
    m = len(y)
    predictions = theta0 + (theta1 * X)
    sq_error = (predictions - y) ** 2
    cost = (1 / (2 * m)) * np.sum(sq_error)
    return cost

def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    """
    Melakukan update parameter theta0 dan theta1 secara iteratif.
    """
    m = len(y)
    cost_history = []
    
    print("\nMemulai Gradient Descent...")
    
    for i in range(iterations):
        predictions = theta0 + (theta1 * X)
        
        error = predictions - y
        
        gradient_0 = (1/m) * np.sum(error)
        gradient_1 = (1/m) * np.sum(error * X)
        
        theta0 = theta0 - (learning_rate * gradient_0)
        theta1 = theta1 - (learning_rate * gradient_1)
        
        cost = compute_cost(X, y, theta0, theta1)
        cost_history.append(cost)
        
        if i % (iterations // 10) == 0:
            print(f"Iterasi {i:5d} | Cost: {cost:.4f} | t0: {theta0:.2f} | t1: {theta1:.2f}")

    return theta0, theta1, cost_history

def main():
    filename = '20489_DiamondDataset.xlsx'
    X, y = load_data(filename)
    
    if X is None:
        return 

    print(f"\nJumlah data: {len(y)}")
    print(f"Range X (Caratage): {np.min(X)} - {np.max(X)}")
    print(f"Range Y (Price): {np.min(y)} - {np.max(y)}")

    print("\n--- Inisialisasi Manual ---")
    try:
        init_theta0 = float(input("Masukkan nilai awal Theta 0 (Intercept/Bias) : "))
        init_theta1 = float(input("Masukkan nilai awal Theta 1 (Slope/Kemiringan): "))
        learning_rate = float(input("Masukkan Learning Rate (alpha) [Saran: 0.1 - 0.01]: "))
        iterations = int(input("Masukkan jumlah iterasi (epochs) [Saran: 1000]       : "))
    except ValueError:
        print("Input harus berupa angka.")
        return

    final_theta0, final_theta1, cost_history = gradient_descent(
        X, y, init_theta0, init_theta1, learning_rate, iterations
    )

    print("\n--- Hasil Akhir ---")
    print(f"Theta 0 Akhir: {final_theta0:.4f}")
    print(f"Theta 1 Akhir: {final_theta1:.4f}")
    print(f"Final MSE Cost : {cost_history[-1]:.4f}")
    print(f"Persamaan Garis: y = {final_theta1:.2f}x + {final_theta0:.2f}")

    regression_line = final_theta0 + (final_theta1 * X)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', label='Data Asli (Diamond)')
    plt.plot(X, regression_line, color='red', linewidth=2, label='Linear Regression')
    plt.title('Linear Regression: Caratage vs Price')
    plt.xlabel('Caratage')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(iterations), cost_history, color='green')
    plt.title('Cost Function (MSE) History')
    plt.xlabel('Iterasi')
    plt.ylabel('Mean Squared Error (Cost)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()