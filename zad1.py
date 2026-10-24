import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt


def wczytaj_obraz(sciezka_lub_url: str) -> np.ndarray:

    #Wczytuje obraz

    if sciezka_lub_url.startswith(("http://", "https://")):
        print(f"Pobieranie obrazu z URL: {sciezka_lub_url} ...")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        req = urllib.request.Request(sciezka_lub_url, headers=headers)
        with urllib.request.urlopen(req) as response:
            image_bytes = response.read()

        # Konwersja surowych bajtów do tablicy i zdekodowanie obrazu
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        obraz = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if obraz is None:
            raise ValueError("Błąd dekodowania obrazu z pobranych danych.")
        return obraz
    else:
        if not os.path.exists(sciezka_lub_url):
            raise FileNotFoundError(f"Nie odnaleziono pliku lokalnego: {sciezka_lub_url}")
        obraz = cv2.imread(sciezka_lub_url)
        if obraz is None:
            raise ValueError(f"Nie udało się wczytać obrazu z pliku: {sciezka_lub_url}")
        return obraz


def main():
    # KROK 1: Wczytanie obrazu
    url_zdalny = "https://upload.wikimedia.org/wikipedia/commons/e/e4/Lemon.jpg"
    plik_lokalny = "lemon.jpg"

    # Jesli plik lokalny istnieje w katalogu roboczym, wczytujemy go w razie pobieramy z sieci
    if os.path.exists(plik_lokalny):
        print(f"KROK 1 Wczytywanie z pliku lokalnego: '{plik_lokalny}'")
        obraz_oryginalny = wczytaj_obraz(plik_lokalny)
    else:
        print(f"KROK 1 Wczytywanie zdalne z adresu: '{url_zdalny}'")
        try:
            obraz_oryginalny = wczytaj_obraz(url_zdalny)
            #zapis lokalnej kopi 
            cv2.imwrite(plik_lokalny, obraz_oryginalny)
            print(f" Obraz zapisano lokalnie jako '{plik_lokalny}'.")
        except Exception as e:
            print(f"Nie udało się pobrać z URL ({e}).")
            obraz_oryginalny = wczytaj_obraz(plik_lokalny)

    wys_oryg, szer_oryg, kanaly = obraz_oryginalny.shape
    print(f"Wymiary początkowe: szerokość={szer_oryg}px, wysokość={wys_oryg}px, kanały={kanaly}")

    # wyświetlenie obrazu oryginalnego z użyciem matplotlib
    obraz_rgb = cv2.cvtColor(obraz_oryginalny, cv2.COLOR_BGR2RGB)

    print("[KROK 2] Wyświetlanie obrazu oryginalnego w osobnym oknie Matplotlib...")
    plt.figure(num="Obraz Oryginalny", figsize=(8, 6))
    plt.title(f"Obraz oryginalny ({szer_oryg}x{wys_oryg} px, RGB)")
    plt.imshow(obraz_rgb)
    plt.xlabel("Współrzędna X (piksele)")
    plt.ylabel("Współrzędna Y (piksele)")
    plt.tight_layout()
    plt.show(block=False)  # block=False pozwala na płynne przejście lub wyświetlenie porównania
    plt.pause(1)

    # Zmniejszenie rozdzielczości o 50% za pomocą cv2.resize
    obraz_zmniejszony = cv2.resize(
        obraz_oryginalny,
        (0, 0),
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA
    )
    wys_zmn, szer_zmn = obraz_zmniejszony.shape[:2]
    print(f"Zmniejszono rozdzielczość o 50%: {szer_zmn}x{wys_zmn} px")

    # Konwersja na odcienie szarości (1 kanał, 0-255)
    obraz_grayscale = cv2.cvtColor(obraz_zmniejszony, cv2.COLOR_BGR2GRAY)
    print(f"Konwersja do grayscale ukończona (nowy kształt: {obraz_grayscale.shape})")

    # cv2.ROTATE_90_CLOCKWISE obraca obraz o 90 stopni zgodnie z ruchem wskazówek zegara
    obraz_obrocony = cv2.rotate(obraz_grayscale, cv2.ROTATE_90_CLOCKWISE)
    wys_obr, szer_obr = obraz_obrocony.shape
    print(f"Obrócono obraz o 90°: nowa szerokość={szer_obr}px, nowa wysokość={wys_obr}px")

  
    # KROK 5: Wyświetlenie obrazu wynikowego oraz porównannie
    print("Wyświetlanie obrazu wynikowego...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), num="Zestawienie obrazów - Zadanie 1")

    # Wykres lewy: oryginał
    axes[0].set_title(f"1. Obraz oryginalny\n({szer_oryg}x{wys_oryg} px, kolor RGB)")
    axes[0].imshow(obraz_rgb)
    axes[0].set_xlabel("X [px]")
    axes[0].set_ylabel("Y [px]")

    # Wykres prawy: wynik po przekształceniach 
    axes[1].set_title(f"2. Obraz przetworzony\n(-50%, grayscale, obrót 90°, {szer_obr}x{wys_obr} px)")
    axes[1].imshow(obraz_obrocony, cmap="gray")
    axes[1].set_xlabel("X [px]")
    axes[1].set_ylabel("Y [px]")

    plt.tight_layout()
    # Zapis do pliku wynikowego jako potwierdzenie wykonania
    plik_wynikowy = "wynik_zad1.png"
    plt.savefig(plik_wynikowy, dpi=150)
    print(f"Wykres wynikowy zapisano do pliku: '{plik_wynikowy}'")
    plt.show()

    # Wyświetlenie macierzy obrazu 
    print("\n" + "=" * 70)
    print("MACIERZ OBRAZU WYNIKOWEGO (TABLICA LICZB):")
    print("=" * 70)
    print(f"Typ tablicy (type)          : {type(obraz_obrocony)}")
    print(f"Typ elementów (dtype)       : {obraz_obrocony.dtype} (wartości 8-bitowe bez znaku, 0-255)")
    print(f"Wymiary macierzy (shape)    : {obraz_obrocony.shape} -> (wiersze/wysokość, kolumny/szerokość)")
    print(f"Liczba wszystkich pikseli   : {obraz_obrocony.size}")
    print(f"Wartość minimalna (czarny)  : {obraz_obrocony.min()}")
    print(f"Wartość maksymalna (biały)  : {obraz_obrocony.max()}")
    print(f"Średnia jasność pikseli     : {obraz_obrocony.mean():.2f}")

    print("\nReprezentacja tablicy NumPy (skrócona z podglądem narożników):")
    print(obraz_obrocony)

    print("\nFragment macierzy (wycinek 10x10 pikseli z lewego górnego rogu):")
    print(obraz_obrocony[:10, :10])
    print("=" * 70)


if __name__ == "__main__":
    main()
