import os
import time
import shutil
import requests
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO, YOLOWorld

# ustawienia treningu
TRENUJ = False
EPOCHS = 30
IMGSZ = 640
BATCH = 16
CONF = 0.25
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FOLDER = os.path.dirname(os.path.abspath(__file__))
DANE = os.path.join(FOLDER, "dane", "data.yaml")
RUNS = os.path.join(FOLDER, "runs")
BEST = os.path.join(FOLDER, "best.pt")
FOLDER_ZDJECIA = os.path.join(FOLDER, "zdjecia")
FOLDER_WYNIKI = os.path.join(FOLDER, "wyniki")

# YOLO-World jako punkt odniesienia, klasy z wczesniejszego eksperymentu
KLASY_WORLD = ["red coca-cola can", "red coca-cola bottle", "coca-cola logo", "blue pepsi can", "pepsi logo"]
CONF_WORLD = 0.15

ZDJECIA = [
    ("coca_cola_butelka.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/e/ef/Coca-Cola_glass_bottle.jpg/1280px-Coca-Cola_glass_bottle.jpg"),
    ("coca_cola_puszka.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/2/2f/Coca-cola_50cl_can_-_Italia.jpg/1280px-Coca-cola_50cl_can_-_Italia.jpg"),
    ("coca_cola_samochod.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/d/d6/Coca-Cola_%C5%A0koda_Praktik_in_Izmail%2C_06.01.2022.jpg/1280px-Coca-Cola_%C5%A0koda_Praktik_in_Izmail%2C_06.01.2022.jpg"),
    ("pepsi_max_puszka.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/b/b6/Pepsi_Max_can.jpg/1280px-Pepsi_Max_can.jpg"),
    ("pepsi_kilka_puszek.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/3/35/Several_Pepsi_Cans.jpg/1280px-Several_Pepsi_Cans.jpg"),
    ("pepsi_diet_puszka.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/d/d0/An_opened_can_of_diet_pepsi_sitting_on_a_table.jpg/1280px-An_opened_can_of_diet_pepsi_sitting_on_a_table.jpg"),
    ("pepsi_reklama_budynek.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/f/fd/Pepsi_LOL_advertisement_on_Hotel_Figueroa.jpg/1280px-Pepsi_LOL_advertisement_on_Hotel_Figueroa.jpg"),
    ("coca_cola_i_pepsi_chiny.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/a7/COCA-COLA_CAN_%26_PEPSI-COLA_CAN_CHINESE_VERSION.jpg/1280px-COCA-COLA_CAN_%26_PEPSI-COLA_CAN_CHINESE_VERSION.jpg"),
]
ILE_Z_VALID = 4

# te ida do wynik_zad3.png
NA_WYNIK = ["coca_cola_i_pepsi_chiny.jpg", "pepsi_kilka_puszek.jpg", "pepsi_reklama_budynek.jpg",
            "pepsi_diet_puszka.jpg", "coca_cola_puszka.jpg", "coca_cola_samochod.jpg"]

KOLOR_COLA = (0, 0, 220)
KOLOR_PEPSI = (200, 80, 0)


def pobierz_zdjecia():
    # bez User-Agent wikimedia zwraca 403
    naglowki = {"User-Agent": "GrafikaLab/1.0 (projekt studencki) python-requests"}
    os.makedirs(FOLDER_ZDJECIA, exist_ok=True)
    for nazwa, url in ZDJECIA:
        sciezka = os.path.join(FOLDER_ZDJECIA, nazwa)
        if os.path.exists(sciezka):
            continue
        print("pobieram", nazwa)
        odp = requests.get(url, headers=naglowki, timeout=60)
        odp.raise_for_status()
        with open(sciezka, "wb") as f:
            f.write(odp.content)
        time.sleep(1)


def trenuj():
    # wagi startowe laduja w runs/ zeby nie smiecic w folderze zadania
    model = YOLO(os.path.join(RUNS, "yolov8n.pt"))
    start = time.time()
    model.train(data=DANE, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                project=RUNS, name="trening", exist_ok=True, amp=False)
    print(f"trening trwal {(time.time() - start) / 60:.1f} min")

    folder_runu = os.path.join(RUNS, "trening")
    shutil.copy(os.path.join(folder_runu, "weights", "best.pt"), BEST)
    for plik in ["results.png", "confusion_matrix.png"]:
        shutil.copy(os.path.join(folder_runu, plik), os.path.join(FOLDER, plik))


def waliduj(model):
    wyniki = model.val(data=DANE, imgsz=IMGSZ, device=DEVICE, project=RUNS, name="walidacja", exist_ok=True)
    print(f"\nmAP50: {wyniki.box.map50:.3f}")
    print(f"mAP50-95: {wyniki.box.map:.3f}")
    print(f"precision: {wyniki.box.mp:.3f}")
    print(f"recall: {wyniki.box.mr:.3f}")
    # na mps czasem wychodzi 0 albo nan
    if wyniki.box.map50 != wyniki.box.map50 or wyniki.box.map50 == 0:
        print("UWAGA: mAP50 wyszlo 0 albo NaN, sprobuj DEVICE = 'cpu'")


def wykryj(model, sciezka, conf, agnostic=False):
    wynik = model.predict(source=sciezka, conf=conf, device=DEVICE, agnostic_nms=agnostic,
                          save=False, verbose=False)[0]
    wykrycia = []
    for box in wynik.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        wykrycia.append((x1, y1, x2, y2, wynik.names[int(box.cls[0])], float(box.conf[0])))
    return wykrycia


def rysuj(obraz, wykrycia):
    grubosc = max(2, obraz.shape[1] // 400)
    skala = max(0.5, obraz.shape[1] / 1300)
    for x1, y1, x2, y2, klasa, pewnosc in wykrycia:
        kolor = KOLOR_COLA if "coca" in klasa else KOLOR_PEPSI
        cv2.rectangle(obraz, (x1, y1), (x2, y2), kolor, grubosc)
        tekst = f"{klasa} {pewnosc:.2f}"
        (tw, th), _ = cv2.getTextSize(tekst, cv2.FONT_HERSHEY_SIMPLEX, skala, grubosc)
        y_tekst = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 6
        cv2.rectangle(obraz, (x1, y_tekst - th - 6), (x1 + tw + 6, y_tekst + 6), kolor, -1)
        cv2.putText(obraz, tekst, (x1 + 3, y_tekst), cv2.FONT_HERSHEY_SIMPLEX, skala, (255, 255, 255), grubosc)
    return obraz


def opis(wykrycia):
    return ", ".join(f"{w[4]} ({w[5]:.2f})" for w in wykrycia) or "nic"


def porownaj(model_world, model_pepsi):
    sciezki = [os.path.join(FOLDER_ZDJECIA, nazwa) for nazwa, _ in ZDJECIA]
    folder_valid = os.path.join(FOLDER, "dane", "valid", "images")
    wszystkie = sorted(os.listdir(folder_valid))
    krok = len(wszystkie) // ILE_Z_VALID
    sciezki += [os.path.join(folder_valid, wszystkie[i * krok]) for i in range(ILE_Z_VALID)]

    os.makedirs(FOLDER_WYNIKI, exist_ok=True)
    pary = {}
    print(f"\n{'zdjecie':<32}| YOLO-World | wytrenowany model")
    for i, sciezka in enumerate(sciezki):
        nazwa = os.path.basename(sciezka)
        if "rf." in nazwa:
            nazwa = f"valid_{i - len(ZDJECIA) + 1}.jpg"
        w_world = wykryj(model_world, sciezka, CONF_WORLD, agnostic=True)
        w_pepsi = wykryj(model_pepsi, sciezka, CONF)
        print(f"{nazwa:<32}| {opis(w_world)} | {opis(w_pepsi)}")

        lewy = rysuj(cv2.imread(sciezka), w_world)
        prawy = rysuj(cv2.imread(sciezka), w_pepsi)
        cv2.imwrite(os.path.join(FOLDER_WYNIKI, nazwa.replace(".jpg", "_porownanie.jpg")), cv2.hconcat([lewy, prawy]))
        pary[nazwa] = (lewy, prawy, len(w_world), len(w_pepsi))

    # siatka: 2 pary w wierszu, lewo YOLO-World, prawo nasz model
    wybrane = NA_WYNIK + ["valid_1.jpg", "valid_2.jpg"]
    wiersze = (len(wybrane) + 1) // 2
    fig, axes = plt.subplots(wiersze, 4, figsize=(20, 5 * wiersze))
    for i, nazwa in enumerate(wybrane):
        lewy, prawy, n1, n2 = pary[nazwa]
        ax1 = axes[i // 2][(i % 2) * 2]
        ax2 = axes[i // 2][(i % 2) * 2 + 1]
        ax1.imshow(cv2.cvtColor(lewy, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"{nazwa}\nYOLO-World: {n1} wykryc", fontsize=10)
        ax2.imshow(cv2.cvtColor(prawy, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"{nazwa}\nwytrenowany (pepsi): {n2} wykryc", fontsize=10)
        ax1.axis("off")
        ax2.axis("off")
    fig.suptitle("YOLO-World (zero-shot) vs YOLOv8n trenowany na logo Pepsi", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(FOLDER, "wynik_zad3.png"), dpi=90)
    plt.close(fig)
    print("\nzapisano wynik_zad3.png i porownania w wyniki/")


def eksport():
    # onnx robimy z kopii w runs/, zeby plik nie wyladowal w folderze zadania
    kopia = os.path.join(RUNS, "eksport", "best.pt")
    os.makedirs(os.path.dirname(kopia), exist_ok=True)
    shutil.copy(BEST, kopia)
    plik = YOLO(kopia).export(format="onnx", imgsz=IMGSZ)
    print("eksport onnx:", plik)


def main():
    print("device:", DEVICE)
    pobierz_zdjecia()

    if TRENUJ or not os.path.exists(BEST):
        trenuj()

    model_pepsi = YOLO(BEST)
    waliduj(model_pepsi)

    model_world = YOLOWorld(os.path.join(RUNS, "yolov8s-world.pt"))
    model_world.set_classes(KLASY_WORLD)

    porownaj(model_world, model_pepsi)
    eksport()


if __name__ == "__main__":
    main()
