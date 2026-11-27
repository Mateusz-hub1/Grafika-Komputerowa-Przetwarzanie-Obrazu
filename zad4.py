import os
import shutil
import time
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ustawienia treningu
TRENUJ = False
EPOCHS = 30
IMGSZ = 640
BATCH = 16
MODEL_START = "yolov8n.pt"
CONF = 0.25
ILE_PREDYKCJI = 6

FOLDER = os.path.dirname(os.path.abspath(__file__))
DANE = os.path.join(FOLDER, "dane")
RUNS = os.path.join(FOLDER, "runs")
NAZWA_RUNU = "trening"
BEST = os.path.join(FOLDER, "best.pt")


def wybierz_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = wybierz_device()


def przygotuj_yaml():
    # kopia yaml z absolutna sciezka, zeby dzialalo z kazdego katalogu
    with open(os.path.join(DANE, "brain-tumor.yaml"), encoding="utf-8") as f:
        linie = f.readlines()
    nowe = []
    for linia in linie:
        if linia.startswith("path:"):
            linia = "path: " + DANE + "\n"
        nowe.append(linia)
    os.makedirs(RUNS, exist_ok=True)
    sciezka = os.path.join(RUNS, "brain-tumor-lokalny.yaml")
    with open(sciezka, "w", encoding="utf-8") as f:
        f.writelines(nowe)
    return sciezka


def policz_zdjecia():
    for czesc in ["train", "val"]:
        folder = os.path.join(DANE, "images", czesc)
        ile = len([p for p in os.listdir(folder) if p.endswith(".jpg")])
        print(f"{czesc}: {ile} zdjec")


def trenuj(sciezka_yaml):
    # wagi startowe laduja w runs/ zeby nie smiecic w folderze zadania
    model = YOLO(os.path.join(RUNS, MODEL_START))
    start = time.time()
    model.train(
        data=sciezka_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=RUNS,
        name=NAZWA_RUNU,
        exist_ok=True,
        amp=False,
        plots=True,
    )
    czas = time.time() - start
    print(f"Trening trwal {czas / 60:.1f} min")
    return os.path.join(RUNS, NAZWA_RUNU, "weights", "best.pt")


def waliduj(sciezka_best, sciezka_yaml):
    model = YOLO(sciezka_best)
    wyniki = model.val(data=sciezka_yaml, imgsz=IMGSZ, device=DEVICE, project=RUNS, name="walidacja", exist_ok=True)
    map50 = wyniki.box.map50
    map50_95 = wyniki.box.map
    print(f"mAP50: {map50:.3f}")
    print(f"mAP50-95: {map50_95:.3f}")
    # na mps czasem wychodzi 0 albo nan, wtedy trzeba puscic na cpu
    if map50 != map50 or map50 == 0:
        print("UWAGA: mAP50 wyszlo 0 albo NaN, sprobuj ustawic DEVICE = 'cpu'")
    return map50, map50_95


def wczytaj_etykiety(sciezka_obrazka, szer, wys):
    # prawdziwe ramki z etykiet (format yolo)
    nazwa = os.path.splitext(os.path.basename(sciezka_obrazka))[0] + ".txt"
    plik = os.path.join(DANE, "labels", "val", nazwa)
    ramki = []
    if not os.path.exists(plik):
        return ramki
    with open(plik) as f:
        for linia in f.read().split("\n"):
            if not linia.strip():
                continue
            _, xc, yc, w, h = [float(x) for x in linia.split()]
            x1 = int((xc - w / 2) * szer)
            y1 = int((yc - h / 2) * wys)
            x2 = int((xc + w / 2) * szer)
            y2 = int((yc + h / 2) * wys)
            ramki.append((x1, y1, x2, y2))
    return ramki


def predykcja(sciezka_best):
    model = YOLO(sciezka_best)
    folder_val = os.path.join(DANE, "images", "val")
    wszystkie = sorted(os.listdir(folder_val))
    krok = len(wszystkie) // ILE_PREDYKCJI
    wybrane = [os.path.join(folder_val, wszystkie[i * krok]) for i in range(ILE_PREDYKCJI)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, sciezka in enumerate(wybrane):
        wynik = model.predict(sciezka, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]
        obraz = cv2.imread(sciezka)
        wys, szer = obraz.shape[:2]

        # zielone - etykieta, czerwone - model
        for x1, y1, x2, y2 in wczytaj_etykiety(sciezka, szer, wys):
            cv2.rectangle(obraz, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for ramka in wynik.boxes:
            x1, y1, x2, y2 = [int(v) for v in ramka.xyxy[0].tolist()]
            pewnosc = float(ramka.conf[0])
            cv2.rectangle(obraz, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(obraz, f"tumor {pewnosc:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ax = axes[i // 3][i % 3]
        ax.imshow(cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{os.path.basename(sciezka)} - wykryto: {len(wynik.boxes)}")
        ax.axis("off")

        if i == 0:
            cv2.imwrite(os.path.join(FOLDER, "przykladowa_predykcja.png"), wynik.plot())

    fig.suptitle("Predykcja YOLOv8n na zdjeciach z valid (zielone - etykieta, czerwone - model)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "wynik_zad4.png"), dpi=120)
    plt.close()
    print("Zapisano wynik_zad4.png i przykladowa_predykcja.png")


def kopiuj_wyniki(sciezka_best):
    folder_runu = os.path.join(RUNS, NAZWA_RUNU)
    for plik in ["results.png", "confusion_matrix.png"]:
        shutil.copy(os.path.join(folder_runu, plik), os.path.join(FOLDER, plik))
    shutil.copy(sciezka_best, os.path.join(FOLDER, "best.pt"))
    print("Skopiowano results.png, confusion_matrix.png i best.pt")


def main():
    print(f"Device: {DEVICE}")
    policz_zdjecia()
    sciezka_yaml = przygotuj_yaml()
    # jak best.pt juz jest to trening pomijamy
    if TRENUJ or not os.path.exists(BEST):
        sciezka_best = trenuj(sciezka_yaml)
        kopiuj_wyniki(sciezka_best)
    else:
        print("Pomijam trening, uzywam best.pt")
    waliduj(BEST, sciezka_yaml)
    predykcja(BEST)


if __name__ == "__main__":
    main()
