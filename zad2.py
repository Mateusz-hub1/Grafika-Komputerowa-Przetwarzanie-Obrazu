import os
import cv2
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# progi do oceny jakosci
PROG_CIEMNY_PIKSEL = 10
PROG_JASNY_PIKSEL = 245
PROG_UDZIAL_CIEMNYCH = 0.10
PROG_UDZIAL_JASNYCH = 0.10
PROG_SREDNIA_CIEMNA = 70
PROG_SREDNIA_JASNA = 185
PROG_ODCHYLENIE = 40
PROG_ZAKRES = 150
PROG_DOMINANTA = 40

USER_AGENT = "GrafikaLab/1.0 (projekt studencki WSB; python-requests)"

# zdjecia z Wikimedia Commons (nazwa, url, czego sie spodziewamy)
ZDJECIA = [
    ("dobre.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/f/f7/Normal_exposure_photography.JPG/960px-Normal_exposure_photography.JPG",
     "dobre"),
    ("ciemne.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/5/54/Underexposed_cat.jpg/960px-Underexposed_cat.jpg",
     "niedoswietlone"),
    ("jasne.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/6/67/Overexposed_photo.JPG/960px-Overexposed_photo.JPG",
     "przeswietlone"),
    ("niebieskie.jpg",
     "https://thumb.wikimedia.org/wikipedia/commons/thumb/7/75/Cold_color_temperature.JPG/960px-Cold_color_temperature.JPG",
     "dominanta niebieska"),
]

FOLDER = os.path.dirname(os.path.abspath(__file__))
FOLDER_ZDJECIA = os.path.join(FOLDER, "zdjecia")
FOLDER_WYNIKI = os.path.join(FOLDER, "wyniki")


def wczytaj_zdjecie(nazwa, url):
    sciezka = os.path.join(FOLDER_ZDJECIA, nazwa)
    if not os.path.exists(sciezka):
        print("Pobieram:", url)
        # bez User-Agent wikimedia zwraca 403
        odp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        odp.raise_for_status()
        with open(sciezka, "wb") as f:
            f.write(odp.content)
    obraz = cv2.imread(sciezka)
    if max(obraz.shape[:2]) > 1200:
        skala = 1200 / max(obraz.shape[:2])
        obraz = cv2.resize(obraz, (0, 0), fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
    return obraz


def policz_histogramy(obraz):
    # histogram jasnosci i kanalow
    szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    hist = {"jasnosc": cv2.calcHist([szary], [0], None, [256], [0, 256]).flatten()}
    for i, kanal in enumerate(["B", "G", "R"]):
        hist[kanal] = cv2.calcHist([obraz], [i], None, [256], [0, 256]).flatten()
    return hist


def statystyki(hist):
    wartosci = np.arange(256)
    stat = {}
    for nazwa, h in hist.items():
        n = h.sum()
        srednia = (h * wartosci).sum() / n
        odchylenie = np.sqrt((h * (wartosci - srednia) ** 2).sum() / n)
        stat[nazwa] = {"srednia": srednia, "odchylenie": odchylenie}

    h = hist["jasnosc"]
    n = h.sum()
    skumulowany = np.cumsum(h) / n
    stat["p1"] = int(np.searchsorted(skumulowany, 0.01))
    stat["p99"] = int(np.searchsorted(skumulowany, 0.99))
    stat["udzial_ciemnych"] = h[:PROG_CIEMNY_PIKSEL + 1].sum() / n
    stat["udzial_jasnych"] = h[PROG_JASNY_PIKSEL:].sum() / n
    return stat


def ocen_jakosc(obraz):
    hist = policz_histogramy(obraz)
    st = statystyki(hist)
    problemy = []

    srednia = st["jasnosc"]["srednia"]
    if st["udzial_ciemnych"] > PROG_UDZIAL_CIEMNYCH or srednia < PROG_SREDNIA_CIEMNA:
        problemy.append("niedoswietlenie (srednia %.0f, %.0f%% pikseli prawie czarnych)"
                        % (srednia, st["udzial_ciemnych"] * 100))
    if st["udzial_jasnych"] > PROG_UDZIAL_JASNYCH or srednia > PROG_SREDNIA_JASNA:
        problemy.append("przeswietlenie (srednia %.0f, %.0f%% pikseli prawie bialych)"
                        % (srednia, st["udzial_jasnych"] * 100))

    zakres = st["p99"] - st["p1"]
    if st["jasnosc"]["odchylenie"] < PROG_ODCHYLENIE or zakres < PROG_ZAKRES:
        problemy.append("niski kontrast (odchylenie %.0f, zakres %d-%d)"
                        % (st["jasnosc"]["odchylenie"], st["p1"], st["p99"]))

    srednie = {k: st[k]["srednia"] for k in ["R", "G", "B"]}
    roznica = max(srednie.values()) - min(srednie.values())
    if roznica > PROG_DOMINANTA:
        kolor = max(srednie, key=srednie.get)
        problemy.append("dominanta koloru %s (roznica srednich kanalow %.0f)" % (kolor, roznica))

    # ocena w skali 1-5, kazdy problem zabiera punkt
    punkty = max(1, 5 - len(problemy))
    if punkty == 5:
        opis = "dobra"
    elif punkty >= 4:
        opis = "srednia"
    else:
        opis = "slaba"
    return punkty, opis, problemy, hist, st


def korekcja_gamma(kanal, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(kanal, lut)


def rozciagnij_histogram(kanal, lo, hi):
    if hi <= lo:
        return kanal
    wynik = (kanal.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return wynik.clip(0, 255).astype(np.uint8)


def clahe(kanal):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(kanal)


def balans_bieli(obraz):
    # gray world
    obraz = obraz.astype(np.float32)
    srednie = obraz.reshape(-1, 3).mean(axis=0)
    obraz = obraz * (srednie.mean() / srednie)
    return obraz.clip(0, 255).astype(np.uint8)


def popraw_zdjecie(obraz, problemy):
    kroki = []
    tekst = " ".join(problemy)

    # jasnosc tylko na kanale L, zeby nie psuc kolorow
    lab = cv2.cvtColor(obraz, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if "niedoswietlenie" in tekst or "przeswietlenie" in tekst:
        gamma = np.log(0.5) / np.log(l.mean() / 255.0)
        gamma = float(np.clip(gamma, 0.3, 3.0))
        l = korekcja_gamma(l, gamma)
        kroki.append("gamma %.2f" % gamma)

    if "niski kontrast" in tekst:
        lo, hi = np.percentile(l, [1, 99])
        if "przeswietlenie" in tekst:
            hi = 255
        l = rozciagnij_histogram(l, lo, hi)
        kroki.append("rozciagniecie histogramu")

    if "niedoswietlenie" in tekst:
        l = clahe(l)
        kroki.append("CLAHE")

    wynik = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # na koniec balans bieli jesli dalej jest dominanta
    st_nowe = statystyki(policz_histogramy(wynik))
    srednie = [st_nowe[k]["srednia"] for k in ["R", "G", "B"]]
    if max(srednie) - min(srednie) > PROG_DOMINANTA:
        wynik = balans_bieli(wynik)
        kroki.append("balans bieli")

    return wynik, kroki


def rysuj_histogram(ax, hist, tytul):
    ax.fill_between(range(256), hist["jasnosc"], color="gray", alpha=0.4, label="jasnosc")
    ax.plot(hist["R"], color="red", linewidth=1, label="R")
    ax.plot(hist["G"], color="green", linewidth=1, label="G")
    ax.plot(hist["B"], color="blue", linewidth=1, label="B")
    ax.set_xlim(0, 255)
    gora = max(hist[k][1:255].max() for k in hist)
    ax.set_ylim(0, gora * 1.15)
    ax.set_title(tytul, fontsize=10)
    ax.set_xlabel("wartosc piksela")
    ax.set_ylabel("liczba pikseli")
    ax.legend(fontsize=7, loc="upper center")


def zapisz_histogramy(obraz, hist, nazwa, tytul):
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    ax[0].imshow(cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB))
    ax[0].set_title(tytul)
    ax[0].axis("off")
    ax[1].bar(range(256), hist["jasnosc"], color="gray", width=1)
    ax[1].set_title("jasnosc (grayscale)")
    for i, (kanal, kolor) in enumerate([("R", "red"), ("G", "green"), ("B", "blue")]):
        ax[i + 2].bar(range(256), hist[kanal], color=kolor, width=1)
        ax[i + 2].set_title("kanal " + kanal)
    for a in ax[1:]:
        a.set_xlim(0, 255)
        a.set_xlabel("wartosc piksela")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_WYNIKI, nazwa), dpi=100)
    plt.close(fig)


def main():
    os.makedirs(FOLDER_ZDJECIA, exist_ok=True)
    os.makedirs(FOLDER_WYNIKI, exist_ok=True)

    wyniki = []
    for nazwa, url, typ in ZDJECIA:
        obraz = wczytaj_zdjecie(nazwa, url)
        punkty, opis, problemy, hist, st = ocen_jakosc(obraz)

        print("\nZdjecie:", nazwa, "(spodziewane: %s)" % typ)
        print("  srednia jasnosc: %.1f, odchylenie: %.1f, zakres p1-p99: %d-%d"
              % (st["jasnosc"]["srednia"], st["jasnosc"]["odchylenie"], st["p1"], st["p99"]))
        print("  srednie R/G/B: %.0f / %.0f / %.0f" % (st["R"]["srednia"], st["G"]["srednia"], st["B"]["srednia"]))
        print("  OCENA: %d/5 (%s)" % (punkty, opis))
        if problemy:
            print("  wykryte problemy:")
            for p in problemy:
                print("   -", p)
        else:
            print("  brak problemow")

        baza = os.path.splitext(nazwa)[0]
        zapisz_histogramy(obraz, hist, "histogram_" + baza + ".png", nazwa)

        # bonus: poprawa zdjecia
        poprawione = None
        if problemy:
            poprawione, kroki = popraw_zdjecie(obraz, problemy)
            punkty2, opis2, problemy2, hist2, st2 = ocen_jakosc(poprawione)
            print("  poprawa:", ", ".join(kroki))
            print("  OCENA PO POPRAWIE: %d/5 (%s)" % (punkty2, opis2))
            for p in problemy2:
                print("   -", p)
            cv2.imwrite(os.path.join(FOLDER_WYNIKI, "poprawione_" + nazwa), poprawione)
            zapisz_histogramy(poprawione, hist2, "histogram_" + baza + "_po.png", nazwa + " po poprawie")
            wyniki.append((nazwa, obraz, hist, punkty, opis, problemy, poprawione, hist2, punkty2, opis2, kroki))
        else:
            wyniki.append((nazwa, obraz, hist, punkty, opis, problemy, None, None, None, None, None))

    # zestawienie na jednym obrazku
    fig, ax = plt.subplots(len(wyniki), 4, figsize=(20, 4.6 * len(wyniki)))
    for i, (nazwa, obraz, hist, punkty, opis, problemy, popr, hist2, punkty2, opis2, kroki) in enumerate(wyniki):
        ax[i, 0].imshow(cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB))
        ax[i, 0].set_title("%s - ocena %d/5 (%s)" % (nazwa, punkty, opis), fontsize=11)
        ax[i, 0].axis("off")
        rysuj_histogram(ax[i, 1], hist, "histogram przed\n" + ("\n".join(p.split(" (")[0] for p in problemy) or "brak problemow"))
        if popr is not None:
            ax[i, 2].imshow(cv2.cvtColor(popr, cv2.COLOR_BGR2RGB))
            ax[i, 2].set_title("po poprawie - ocena %d/5 (%s)\n%s" % (punkty2, opis2, ", ".join(kroki)), fontsize=11)
            ax[i, 2].axis("off")
            rysuj_histogram(ax[i, 3], hist2, "histogram po poprawie")
        else:
            ax[i, 2].text(0.5, 0.5, "zdjecie dobre,\nbez poprawy", ha="center", va="center", fontsize=14)
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "wynik_zad2.png"), dpi=90)
    plt.close(fig)
    print("\nZapisano wynik_zad2.png i obrazki w folderze wyniki/")


if __name__ == "__main__":
    main()
