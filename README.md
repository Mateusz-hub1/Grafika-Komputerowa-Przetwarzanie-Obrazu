# Zadanie 4 - trenowanie własnego modelu YOLO

Wykrywanie guzów mózgu na skanach MRI. Model YOLOv8n (Ultralytics) został douczony na małym zbiorze `brain-tumor`.

Model jest tylko do nauki na zajęcia i nie nadaje się do diagnozy medycznej.

## Zbiór danych

- Źródło: zbiór Brain Tumor od Ultralytics, https://docs.ultralytics.com/datasets/detect/brain-tumor/
- Licencja: AGPL-3.0 (plik `dane/LICENSE.txt`)
- Archiwum ma ok. 4 MB, po rozpakowaniu ok. 11 MB.
- Kopia zbioru jest w folderze `dane/` (obrazki, etykiety w formacie YOLO, `brain-tumor.yaml`), więc trening nie wymaga pobierania danych.
- Pobrana wersja yaml ma **jedną klasę**: `0: tumor` (starsza wersja zbioru miała dwie klasy negative/positive, ta już nie).
- Zbiór nie ma części `test`, są tylko `train` i `val`.

| Część | Zdjęcia | Ramki (guzy) |
|-------|---------|--------------|
| train | 994     | 1057         |
| val   | 121     | 123          |

## Uruchomienie

```bash
pip install -r requirements_zad4.txt
python zad4.py
```

Skrypt działa z dowolnego katalogu (ścieżki liczone od położenia pliku). Kolejno:

1. tworzy w `runs/` kopię `dane/brain-tumor.yaml` z absolutną ścieżką do `dane/`,
2. trenuje `yolov8n.pt` (wagi startowe pobierają się same do `runs/`),
3. robi walidację i wypisuje mAP50 oraz mAP50-95,
4. kopiuje `results.png`, `confusion_matrix.png` i `best.pt` do tego folderu,
5. robi predykcję na 6 zdjęciach z `val` i zapisuje `wynik_zad4.png` oraz `przykladowa_predykcja.png`.

Ustawienia są stałymi na górze pliku: `EPOCHS = 30`, `IMGSZ = 640`, `BATCH = 16`, `CONF = 0.25`. Urządzenie wybierane jest automatycznie: `mps` na Macu z Apple Silicon, w przeciwnym razie `cpu`. Jeśli na `mps` mAP50 wyjdzie 0 albo NaN, skrypt wypisze ostrzeżenie i wtedy trzeba ustawić `DEVICE = "cpu"`. W moim przypadku na `mps` wszystko działało poprawnie.

Folder `runs/` z pełnymi wynikami treningu nie jest dodawany do repozytorium.

## Wyniki

| Parametr        | Wartość                     |
|-----------------|-----------------------------|
| Model startowy  | yolov8n.pt                  |
| Klasy           | 1 (`tumor`)                 |
| Zdjęcia         | 994 train, 121 val          |
| Epoki           | 30                          |
| Rozmiar obrazu  | 640                         |
| Urządzenie      | mps (Apple M2)              |
| Czas treningu   | ok. 26 min                  |
| mAP50           | 0.988                       |
| mAP50-95        | 0.747                       |

Na wykresach w `results.png` straty spadają przez cały trening, a mAP50 rośnie szybko i od ok. 15. epoki stoi w okolicach 0.97-0.99. Dalsze epoki poprawiają głównie mAP50-95, czyli dokładność dopasowania ramek.

Macierz pomyłek z `confusion_matrix.png` jest liczona przy bardzo niskim progu pewności używanym w walidacji, dlatego widać tam dużo fałszywych wykryć na tle (141). Przy progu `conf = 0.25` (tym samym co w predykcji) model znalazł 119 z 123 guzów, pominął 4 i dał 11 fałszywych wykryć.

Na `wynik_zad4.png` zielone ramki to etykiety ze zbioru, czerwone to wykrycia modelu z pewnością. Większość guzów jest trafiona dokładnie, czasem model zaznacza dodatkowo drugi, podobnie wyglądający obszar albo dubluje ramkę.

## Pliki

- `zad4.py` - trening, walidacja i predykcja
- `dane/` - zbiór danych (obrazki, etykiety, yaml, licencja)
- `best.pt` - najlepsze wagi po treningu (ok. 6.5 MB)
- `results.png` - wykresy strat i metryk z treningu
- `confusion_matrix.png` - macierz pomyłek
- `przykladowa_predykcja.png` - predykcja na jednym zdjęciu
- `wynik_zad4.png` - predykcja na 6 zdjęciach z `val`
- `requirements_zad4.txt` - potrzebne biblioteki
