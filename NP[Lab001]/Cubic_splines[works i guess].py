import requests
import numpy as np
import matplotlib.pyplot as plt

def get_route_data():
    raw_coords = [
        "48.164214,24.536044", "48.164983,24.534836", "48.165605,24.534068",
        "48.166228,24.532915", "48.166777,24.531927", "48.167326,24.530884",
        "48.167011,24.530061", "48.166053,24.528039", "48.166655,24.526064",
        "48.166497,24.523574", "48.166128,24.520214", "48.165416,24.517170",
        "48.164546,24.514640", "48.163412,24.512980", "48.162331,24.511715",
        "48.162015,24.509462", "48.162147,24.506932", "48.161751,24.504244",
        "48.161197,24.501793", "48.160580,24.500537", "48.160250,24.500106"
    ]

    locations_str = "|".join(raw_coords)
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations_str}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data["results"]

    except Exception as e:
        print("API failed:", e)

        results = []
        start_alt = 1250
        end_alt = 2061
        n = len(raw_coords)

        for i, coord in enumerate(raw_coords):
            lat, lon = map(float, coord.split(','))
            elev = start_alt + (end_alt - start_alt) * (i / (n - 1))
            results.append({
                "latitude": lat,
                "longitude": lon,
                "elevation": elev
            })

        return results
    
results = get_route_data()
n = len(results)

print("Кількість вузлів:", n)
print("№ | Latitude | Longitude | Elevation (m)")
for i, p in enumerate(results):
    print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")

def haversine(lat1, lon1, lat2, lon2): # -> dist between pts
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

distances = [0] # -> tab
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print("№ | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

filename = "route_data.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write("Route: Zarosliak -> Hoverla\n")

    f.write("Raw API data:\n")
    f.write("№ | Latitude | Longitude | Elevation (m)\n")
    f.write("-----------------------------------------\n")
    for i, p in enumerate(results):
        f.write(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}\n")

    f.write("\nTabulation (Cumulative distance vs Elevation):\n")
    f.write("№ | Distance (m) | Elevation (m)\n")
    f.write("-----------------------------------------\n")
    for i in range(n):
        f.write(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}\n")

x_points = np.array(distances)
y_points = np.array(elevations)

plt.figure(figsize=(10,6))
plt.plot(x_points, y_points, 'o-', label="Discrete nodes")
plt.xlabel("Відстань(m)")
plt.ylabel("Висота (m)")
plt.title("графік залежності кумулятивної відстані до висоти Заросляк -> Говерла")
plt.grid(True)
plt.legend()
plt.show()

def build_tridiagonal_system(x, y):
    n = len(x)
    h = np.diff(x)

    size = n - 2

    lower = np.zeros(size-1)
    main = np.zeros(size)
    upper = np.zeros(size-1)
    rhs = np.zeros(size)

    for i in range(1, n-1):
        hi_prev = h[i-1]
        hi = h[i]

        if i != 1:
            lower[i-2] = hi_prev

        main[i-1] = 2 * (hi_prev + hi)

        if i != n-2:
            upper[i-1] = hi

        rhs[i-1] = 3 * (
            (y[i+1] - y[i]) / hi
            - (y[i] - y[i-1]) / hi_prev
        )

    return lower, main, upper, rhs

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0]/b[0]
    d_prime[0] = d[0]/b[0]

    for i in range(1, n-1):
        temp = b[i] - a[i-1]*c_prime[i-1]
        c_prime[i] = c[i]/temp
        d_prime[i] = (d[i] - a[i-1]*d_prime[i-1]) / temp

    d_prime[n-1] = (d[n-1] - a[n-2]*d_prime[n-2]) / (b[n-1] - a[n-2]*c_prime[n-2])

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in reversed(range(n-1)):
        x[i] = d_prime[i] - c_prime[i]*x[i+1]

    return x

a, b, c, d = build_tridiagonal_system(x_points, y_points)

lower, main, upper, rhs = build_tridiagonal_system(x_points, y_points)

print("\nTridiagonal system coefficients:")
print("Lower diagonal:", lower)
print("Main diagonal:", main)
print("Upper diagonal:", upper)
print("Right-hand side:", rhs)

C_internal = thomas_algorithm(lower, main, upper, rhs)

def compute_spline_coefficients(x, y, C):
    n = len(x)
    h = np.diff(x)

    a = y[:-1]
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    c = C[:-1]

    for i in range(n-1):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2*C[i] + C[i+1]) / 3
        d[i] = (C[i+1] - C[i]) / (3*h[i])

    return a, b, c, d

C = np.zeros(n)
C[1:-1] = C_internal

C_full = np.zeros(len(x_points))
C_full[1:-1] = C_internal

a_full, b_full, c_full, d_full = compute_spline_coefficients(
    x_points, y_points, C_full
)

print("\nКоефіцієнти кубічних спрайнів(21 вузлів):")
for i in range(len(a_full)):
    print(f"\n Сегмент сплайну {i}:")
    print(f"a[{i}] = {a_full[i]}")
    print(f"b[{i}] = {b_full[i]}")
    print(f"c[{i}] = {c_full[i]}")
    print(f"d[{i}] = {d_full[i]}")

x_dense = np.linspace(x_points[0], x_points[-1], 1000)
y_reference = np.zeros_like(x_dense)

for i in range(len(a_full)):
    mask = (x_dense >= x_points[i]) & (x_dense <= x_points[i+1])
    dx = x_dense[mask] - x_points[i]
    y_reference[mask] = (
        a_full[i]
        + b_full[i]*dx
        + c_full[i]*dx**2
        + d_full[i]*dx**3
    )

print("\nSolution C_i (second derivatives):")
for i in range(n):
    print(f"C[{i}] = {C[i]}")

def compute_error_for_nodes(num_nodes):
    idx = np.linspace(0, len(x_points)-1, num_nodes, dtype=int)
    x_sub = x_points[idx]
    y_sub = y_points[idx]

    lower, main, upper, rhs = build_tridiagonal_system(x_sub, y_sub)
    C_internal = thomas_algorithm(lower, main, upper, rhs)

    C = np.zeros(len(x_sub))
    C[1:-1] = C_internal

    a, b, c, d = compute_spline_coefficients(x_sub, y_sub, C)

    y_spline = np.zeros_like(x_dense)

    for i in range(len(a)):
        mask = (x_dense >= x_sub[i]) & (x_dense <= x_sub[i+1])
        dx = x_dense[mask] - x_sub[i]
        y_spline[mask] = (
            a[i]
            + b[i]*dx
            + c[i]*dx**2
            + d[i]*dx**3
        )

    return np.abs(y_reference - y_spline)

def interpolate_and_plot(num_nodes):
    idx = np.linspace(0, len(x_points)-1, num_nodes, dtype=int)
    x_sub = x_points[idx]
    y_sub = y_points[idx]

    lower, main, upper, rhs = build_tridiagonal_system(x_sub, y_sub)
    C_internal = thomas_algorithm(lower, main, upper, rhs)

    C = np.zeros(len(x_sub))
    C[1:-1] = C_internal

    a, b, c, d = compute_spline_coefficients(x_sub, y_sub, C)

    x_dense = np.linspace(x_sub[0], x_sub[-1], 1000)
    y_dense = np.zeros_like(x_dense)

    for i in range(len(a)):
        mask = (x_dense >= x_sub[i]) & (x_dense <= x_sub[i+1])
        dx = x_dense[mask] - x_sub[i]
        y_dense[mask] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    plt.plot(x_dense, y_dense, label=f"{num_nodes} вузлів")

def print_error_stats(num_nodes, error_array):
    avg_error = np.mean(error_array)
    max_error = np.max(error_array)

    print(f"\n{num_nodes} вузлів:")
    print(f"Середня похибка: {avg_error:.6f}")
    print(f"Максимальна похибка: {max_error:.6f}")

plt.figure(figsize=(8,6))
interpolate_and_plot(21)
interpolate_and_plot(10)
interpolate_and_plot(15)
interpolate_and_plot(20)

plt.legend()
plt.title("Вплив кількості вузлів")
plt.show()

y_spline = y_reference.copy()

error_10 = compute_error_for_nodes(10)
error_15 = compute_error_for_nodes(15)
error_20 = compute_error_for_nodes(20)

plt.figure(figsize=(10,6))
plt.plot(x_dense, error_10, label="10 вузлів")
plt.plot(x_dense, error_15, label="15 вузлів")
plt.plot(x_dense, error_20, label="20 вузлів")

print_error_stats(10, error_10)
print_error_stats(15, error_15)
print_error_stats(20, error_20)

plt.legend()
plt.title("Похибка апкроксимації")
plt.xlabel("Distance (m)")
plt.ylabel("|y - y_aprx|")
plt.grid(True)
plt.show()