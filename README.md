# Zadanie 3 – detekcja logo Pepsi

Porównanie dwóch podejść do wykrywania logo na zdjęciach:

1. **YOLO-World** (`yolov8s-world.pt`) – gotowy model bez trenowania (zero-shot), klasy podawane tekstem przez `set_classes([...])`. To punkt odniesienia z pierwszej wersji zadania.
2. **YOLOv8n trenowany na logo Pepsi** – model `yolov8n.pt` dostrojony na zbiorze z Roboflow Universe, według przepisu z materiałów do zajęć (`YOLO("yolov8n.pt")` → `model.train` → `model.val` → `model.predict` → `model.export`).

Pierwotnie model miał rozróżniać logo Pepsi i Coca-Coli. W znalezionym zbiorze z Coca-Colą ramki obejmowały całe butelki zamiast samego logo, więc nie pasował do zadania i ostatecznie model jest trenowany tylko na jednej klasie `pepsi`.

## Jak uruchomić

```bash
pip install -r requirements_zad3.txt
python zad3.py
```

Skrypt działa z dowolnego katalogu (ścieżki liczone są względem pliku `zad3.py`). Kolejne kroki:

1. pobiera zdjęcia testowe z Wikimedia Commons do `zdjecia/` (jeśli ich nie ma),
2. trenuje model, ale tylko gdy `TRENUJ = True` albo gdy nie ma pliku `best.pt` (w repo `best.pt` już jest, więc domyślnie trening jest pomijany),
3. waliduje `best.pt` na części `valid` (mAP50, mAP50-95, precision, recall),
4. uruchamia oba modele na tych samych zdjęciach i zapisuje porównania do `wyniki/` oraz `wynik_zad3.png`,
5. eksportuje model do formatu ONNX.

Ustawienia są stałymi na górze pliku: `EPOCHS = 30`, `IMGSZ = 640`, `BATCH = 16`, `CONF = 0.25`, urządzenie `mps` (Mac z Apple Silicon), a jak go nie ma, to `cpu`.

Wagi startowe `yolov8n.pt` i `yolov8s-world.pt` pobierają się do `runs/`. Przy pierwszym użyciu YOLO-World `ultralytics` pobiera też model CLIP (ok. 340 MB) do podfolderu `weights/` w bieżącym katalogu. Folder `runs/` i pliki `*.pt` (poza `best.pt`) nie są w repozytorium.

## Zbiór danych

- Źródło: [Pepsi Logo Detection](https://universe.roboflow.com/detectionanas/pepsi-logo-detection/dataset/1), Roboflow Universe, wersja 1
- Autor: detectionanas
- Licencja: CC BY 4.0
- Pobrany ręcznie jako zip w formacie YOLOv8 i rozpakowany do `dane/`.
- 1 klasa: `pepsi` (ramki obejmują samo logo, czyli charakterystyczną kulę Pepsi)
- 1559 zdjęć: 1241 w `train` (2045 ramek) i 318 w `valid` (508 ramek). Zbiór nie ma części `test`.

Zmiany względem oryginału:

- zdjęcia zmniejszone z 1280x1280 do 640x640 (trening i tak idzie na `imgsz=640`, a zbiór zajmuje mniej miejsca w repo). Etykiety YOLO są we współrzędnych względnych, więc się nie zmieniły;
- poprawiony `data.yaml`: ścieżki względem pliku (`train/images`, `valid/images`) i usunięta nieistniejąca część `test`.

Część etykiet jest zapisana jako wielokąty (format segmentacji), `ultralytics` przy treningu detekcji sam zamienia je na prostokątne ramki.

## Wyniki treningu

| Parametr | Wartość |
|---|---|
| Model startowy | `yolov8n.pt` |
| Klasy | 1 (`pepsi`) |
| Zdjęcia train / valid | 1241 / 318 |
| Epoki | 30 |
| imgsz / batch | 640 / 16 |
| Urządzenie | mps (Apple M2) |
| Czas treningu | ok. 35 min |
| **mAP50** | **0.924** |
| mAP50-95 | 0.686 |
| Precision | 0.827 |
| Recall | 0.909 |

Model ma tylko jedną klasę, więc wynik dla `pepsi` jest jednocześnie wynikiem całego modelu. Na `mps` metryki wyszły poprawne (nie było zer ani NaN), więc nie było potrzeby trenować na `cpu`.

Krzywe uczenia są w `results.png`, a macierz pomyłek w `confusion_matrix.png`.

## Porównanie YOLO-World i wytrenowanego modelu

YOLO-World dostaje klasy `red coca-cola can`, `red coca-cola bottle`, `coca-cola logo`, `blue pepsi can`, `pepsi logo` (najlepszy zestaw z poprzedniego eksperymentu, `conf = 0.15`, `agnostic_nms`). Wytrenowany model ma tylko klasę `pepsi` (`conf = 0.25`).

| Zdjęcie | YOLO-World | Wytrenowany model (pepsi) |
|---|---|---|
| `coca_cola_butelka.jpg` (butelka Coca-Coli) | red coca-cola bottle 0.95 | nic |
| `coca_cola_puszka.jpg` (puszka Coca-Coli) | red coca-cola bottle 0.86 (to puszka) | nic |
| `coca_cola_samochod.jpg` (logo Coca-Coli na aucie) | nic | nic |
| `pepsi_max_puszka.jpg` (czarna puszka Pepsi Max) | blue pepsi can 0.84 | pepsi 0.84 |
| `pepsi_kilka_puszek.jpg` (5 puszek Pepsi Max Cherry) | 5 × blue pepsi can (0.43–0.92) | 2 × pepsi (0.83, 0.81) |
| `pepsi_diet_puszka.jpg` (Diet Pepsi) | blue pepsi can 0.85 | pepsi 0.88 |
| `pepsi_reklama_budynek.jpg` (reklama na budynku) | nic | pepsi 0.89 |
| `coca_cola_i_pepsi_chiny.jpg` (obie puszki) | blue pepsi can 0.93, red coca-cola bottle 0.89 | pepsi 0.88 (tylko na puszce Pepsi) |
| `valid_1.jpg` (reklama Pepsi, 4 logo) | red coca-cola bottle 0.56, pepsi logo 0.16 | 4 × pepsi (0.66–0.89) |
| `valid_2.jpg` (billboardy na ulicy) | nic | 2 × pepsi (0.77, 0.66) |
| `valid_3.jpg` | nic | nic |
| `valid_4.jpg` | nic | 3 × pepsi (0.34–0.77) |

Zdjęcia `valid_1`–`valid_4` to co któreś zdjęcie z `dane/valid` (model ich nie widział podczas treningu, ale pochodzą z tego samego zbioru). Porównania każdego zdjęcia (lewa strona YOLO-World, prawa wytrenowany model) są w `wyniki/*_porownanie.jpg`, a 8 wybranych zestawionych jest w `wynik_zad3.png`.

### Zdjęcia z Coca-Colą

Model był uczony tylko na logo Pepsi, więc na Coca-Coli nie powinien niczego wykrywać i tak się stało: na butelce, puszce i samochodzie z logo Coca-Coli nie ma żadnej ramki, a na zdjęciu z dwiema puszkami zaznaczył tylko puszkę Pepsi. YOLO-World w tych samych miejscach zwraca ramki `red coca-cola bottle`, bo ma osobną klasę dla Coli.

## Wnioski

- Wytrenowany model wykrywa **samo logo** (kulę Pepsi), a YOLO-World zaznacza **cały produkt** i rozpoznaje markę głównie po kolorze puszki. Widać to dobrze na reklamie na budynku: YOLO-World nie znajduje nic, bo nie ma tam puszki, a wytrenowany model łapie logo z pewnością 0.89.
- Na zdjęciach w stylu zbioru treningowego (reklamy, billboardy, kilka małych logo) wytrenowany model jest wyraźnie lepszy. YOLO-World na `valid_1` zaznaczył szklankę z colą jako `red coca-cola bottle`.
- Wytrenowany model nie jest idealny: na zdjęciu z pięcioma puszkami Pepsi Max Cherry znalazł tylko 2 logo. Na tych puszkach jest nowe logo Pepsi z 2023 roku (napis PEPSI na kuli), a zbiór pochodzi z 2022 roku, czyli sprzed zmiany logo. Do tego część puszek jest obrócona albo zasłonięta. Na `valid_3` nie znalazł nic, a na `valid_4` są wykrycia z niską pewnością (0.34–0.35).
- YOLO-World ma tę zaletę, że działa od razu i bez danych, ale do konkretnego logo lepszy jest nawet mały model `yolov8n` dostrojony na ponad tysiącu zdjęć.
- Zdjęcia z `valid` pochodzą z tego samego źródła co treningowe, więc na zupełnie innych zdjęciach mAP byłby pewnie niższy.

## Eksport do ONNX

Na końcu skrypt wywołuje `model.export(format="onnx")` na kopii `best.pt` w `runs/eksport/`. Powstaje plik `runs/eksport/best.onnx` (ok. 12 MB), który można uruchomić np. przez `onnxruntime` albo `YOLO("best.onnx")`. Pliku `.onnx` nie ma w repozytorium, bo można go w każdej chwili wygenerować ponownie.

## Pliki

- `zad3.py` – trening, walidacja, porównanie z YOLO-World, eksport
- `dane/` – zbiór Pepsi Logo Detection (zdjęcia 640x640, etykiety YOLO, `data.yaml`, opis licencji)
- `best.pt` – wagi wytrenowanego modelu
- `results.png`, `confusion_matrix.png` – wykresy z treningu
- `zdjecia/` – zdjęcia testowe z Wikimedia Commons
- `wyniki/` – porównania obu modeli dla każdego zdjęcia
- `wynik_zad3.png` – zestawienie wybranych porównań
- `requirements_zad3.txt` – potrzebne biblioteki

## Źródła zdjęć testowych

Wszystkie z Wikimedia Commons (wersje 1280 px):

| Plik | Źródło |
|---|---|
| `coca_cola_butelka.jpg` | [Coca-Cola_glass_bottle.jpg](https://commons.wikimedia.org/wiki/File:Coca-Cola_glass_bottle.jpg) |
| `coca_cola_puszka.jpg` | [Coca-cola_50cl_can_-_Italia.jpg](https://commons.wikimedia.org/wiki/File:Coca-cola_50cl_can_-_Italia.jpg) |
| `coca_cola_samochod.jpg` | [Coca-Cola Škoda Praktik in Izmail](https://commons.wikimedia.org/wiki/File:Coca-Cola_%C5%A0koda_Praktik_in_Izmail,_06.01.2022.jpg) |
| `pepsi_max_puszka.jpg` | [Pepsi_Max_can.jpg](https://commons.wikimedia.org/wiki/File:Pepsi_Max_can.jpg) |
| `pepsi_kilka_puszek.jpg` | [Several_Pepsi_Cans.jpg](https://commons.wikimedia.org/wiki/File:Several_Pepsi_Cans.jpg) |
| `pepsi_diet_puszka.jpg` | [An opened can of diet pepsi](https://commons.wikimedia.org/wiki/File:An_opened_can_of_diet_pepsi_sitting_on_a_table.jpg) |
| `pepsi_reklama_budynek.jpg` | [Pepsi LOL advertisement](https://commons.wikimedia.org/wiki/File:Pepsi_LOL_advertisement_on_Hotel_Figueroa.jpg) |
| `coca_cola_i_pepsi_chiny.jpg` | [COCA-COLA CAN & PEPSI-COLA CAN](https://commons.wikimedia.org/wiki/File:COCA-COLA_CAN_%26_PEPSI-COLA_CAN_CHINESE_VERSION.jpg) |
