# -*- coding: utf-8 -*-
"""
October 2022
    serie czasowe
"""


# Robiłem opisy funkcji, których zastosowałem tylko przy osobach zdrowych. Dla osób chorych jest to identyczna metoda,
# której nie chciałem pisać drugi raz.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA




# ustawiamy katalogi pracy
import os
KATALOG_PROJEKTU = os.path.join(os.getcwd(), "hypertension")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU, "dane")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "wykresy")
os.makedirs(KATALOG_WYKRESOW, exist_ok = True)
os.makedirs(KATALOG_DANYCH, exist_ok = True)


# PROSZĘ WPISAĆ SWOJĄ ŚCIEŻKĘ

SKAD_POBIERAC = "C:/Users/User/hypertension/dane"

czytamy = SKAD_POBIERAC
print ('\nPrzetwarzamy katalog', czytamy)

#%%
# Zmieniamy nazwę pliku ponieważ zawiera spację:
    
#old_name = r"C:\Users\User\hypertension\dane\Jer_ 1A_027.txt"
#new_name = r"C:\Users\User\hypertension\dane\Jer_1A_027.txt"

#os.rename(old_name, new_name)


#%%

pliki = os.listdir(czytamy)
[print(pliki.index(item) , ':', item)  for item in pliki]


#do_usuniecia = ["Iwo_1A_020.txt", "Jan_1A_025.txt", "Jer_1A_027.txt", "Jer_1A_014.txt", "Joa_1A_012.txt",
#                "Jol_1A_023.txt", "Paw_1A_013.txt", "Rom_1A_030.txt", "Zbi_2A_016.txt"]

#for i in do_usuniecia:
 #   pliki.remove(i)

#while (True):
#   try:
#       num_plik = int(input('ktory plik? '))
#       print('przerabiamy ', pliki[num_plik])
#   except  ValueError: 
#       print('zly numer pliku ')
#   else:
#       break
#%%

num_plik = 0
plik = pliki[num_plik]
print('przerabiamy ', plik)



def load_serie(skad , co, ile_pomin = 0, kolumny = ['RR_interval', 'Num_contraction']):
    csv_path = os.path.join(skad, co)
    print ( skad, co)
    seria = pd.read_csv(csv_path, sep = '\t', header = None,
                        skiprows = ile_pomin, names = kolumny, encoding = 'latin1')
    if skad == SKAD_POBIERAC[2]:
        seria = pd.read_csv(csv_path, sep = '\t',  decimal = ',' )
    return seria


pomin = 5
kolumny = ['time','R_peak','Resp','SBP','cus']


#%%Tworzymy listę zdorwych i chorych osób

lista_zdrowych = []
lista_chorych = []

# Osoby, które mają problem z nadciśnieniem mają w 5 znaku nazwy cyferkę "2", natomiast osoby zdrowe mają "1".
# Dzielimy pacjentów na zdrowych i chorych

for i in pliki:
    if i[4] == "1":
        lista_zdrowych.append(i)

for j in pliki:
    if j[4] == "2":
        lista_chorych.append(j)

#%%

# Teraz wgrywamy listę dataframe'ów tych osób

tabele_zdrowych = []
tabele_chorych = []

for i in lista_zdrowych:
    serie = load_serie(skad = czytamy, co = i, ile_pomin = pomin, kolumny = kolumny)
    tabele_zdrowych.append(serie)

for j in lista_chorych:
    series = load_serie(skad = czytamy, co = j, ile_pomin = pomin, kolumny = kolumny)
    tabele_chorych.append(series)

#print(tabele_zdrowych)
#print(tabele_chorych)

#%%
'''
def find_rows_with_letter(df, column_name, letters):
    rows_with_letter = []
    for index, row in df.iterrows():
        if isinstance(row[column_name], str) and any(letter in row[column_name] for letter in letters):
            rows_with_letter.append(index)
    return rows_with_letter

target_letters = ["a", "o", "e", "y", "i", "u"]

for i, df in enumerate(tabele_zdrowych):
    column_name = 'time'
    rows_with_letter = find_rows_with_letter(df, column_name, target_letters)
    print(f"DataFrame {i + 1} - Rows with '{target_letters}' in '{column_name}':")
    print(rows_with_letter)

for i, df in enumerate(tabele_chorych):
    column_name = 'time'
    rows_with_letter = find_rows_with_letter(df, column_name, target_letters)
    print(f"DataFrame {i + 1} - Rows with '{target_letters}' in '{column_name}':")
    print(rows_with_letter)

'''
#%%

w = [35550, 35551, 35552, 201755, 201756, 201757, 207710, 207711, 207712]
tabele_zdrowych[7] = tabele_zdrowych[7].drop(w)
tabele_zdrowych[7].reset_index(drop = True, inplace = True)

w = [7000, 7001, 7002]
tabele_zdrowych[11] = tabele_zdrowych[11].drop(w)
tabele_zdrowych[11].reset_index(drop = True, inplace = True)

w = [20000, 20001, 20002, 42305, 42306, 42307, 47210, 47211, 47212]
tabele_zdrowych[13] = tabele_zdrowych[13].drop(w)
tabele_zdrowych[13].reset_index(drop = True, inplace = True)

w = [7000, 7001, 7002]
tabele_zdrowych[14] = tabele_zdrowych[14].drop(w)
tabele_zdrowych[14].reset_index(drop = True, inplace = True)

w = [19300, 19301, 19302, 42505, 42506, 42507, 58910, 58911, 58912, 71265, 71266, 71267]
tabele_zdrowych[15] = tabele_zdrowych[15].drop(w)
tabele_zdrowych[15].reset_index(drop = True, inplace = True)

w = [7000, 7001, 7002]
tabele_zdrowych[17] = tabele_zdrowych[17].drop(w)
tabele_zdrowych[17].reset_index(drop = True, inplace = True)

w = [43450, 43451, 43452, 48005, 48006, 48007]
tabele_zdrowych[21] = tabele_zdrowych[21].drop(w)
tabele_zdrowych[21].reset_index(drop = True, inplace = True)

w = [7000, 7001, 7002, 12055, 12056, 12057]
tabele_zdrowych[23] = tabele_zdrowych[23].drop(w)
tabele_zdrowych[23].reset_index(drop = True, inplace = True)

w = [223650, 223651, 223652]
tabele_chorych[36] = tabele_chorych[36].drop(w)
tabele_chorych[36].reset_index(drop = True, inplace = True)

#%%
indexes_to_process = [7, 11, 13, 14, 15, 17, 21, 23]

for index in indexes_to_process:
    i = tabele_zdrowych[index]
    i["R_peak"].replace("0", 0, inplace=True)
    i["R_peak"].replace('', 0, inplace=True)
    i["R_peak"].replace('1', 1, inplace=True)
    i["time"] = i["time"].astype(float)
    i["Resp"] = i["Resp"].astype(float)
    i["SBP"] = i["SBP"].astype(float)

#%%
indexes_to_process = [36]

for index in indexes_to_process:
    i = tabele_chorych[index]
    i["R_peak"].replace("0", 0, inplace=True)
    i["R_peak"].replace('', 0, inplace=True)
    i["R_peak"].replace('1', 1, inplace=True)
    i["time"] = i["time"].astype(float)
    i["Resp"] = i["Resp"].astype(float)
    i["SBP"] = i["SBP"].astype(float)

#%% Zczytujemy nazwy plików, tak by pozbyć się końcówki ".txt". Nadajemy nazwy pacjentom, aby ich odróżniać.

names_zdrowych = []
names_chorych = []

for i in lista_zdrowych:
    x = i[:10]
    names_zdrowych.append(x)

for i in lista_chorych:
    x = i[:10]
    names_chorych.append(x)

#print(names_zdrowych)
#print(names_chorych)

#%%

seria = load_serie(skad = czytamy, co = plik, ile_pomin = pomin, kolumny = kolumny)
print(seria.info())
seria = pd.DataFrame(seria)


#%%
# ogólna informacja kompletnoci danych
print("\n ogólna informacja o kompletnosci danych")
print('\n-->brak danych w danej kolumnie:\n')

print(seria.isnull().sum())
print(seria.describe())

#%% kwantowanie
print(seria['R_peak'].value_counts().sort_index())

print(seria["R_peak"][7])

#%% wizualizacja danych : DataFrame.plot

seria.hist(bins = 50, figsize = (9, 6))
plt.tight_layout()
plt.title("histogramy wartosći")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogramy.jpg'), dpi = 300 ) 
plt.show() # sprawdzić jak się plik zapisal

#%% Obliczamy interwały RR dla chorych i zdrowych osób

#Interwał RR  - czas pomiędzy wierzchołkami R na krzywej EKG.
#SDNN - odchylenie standardowe interwałów RR
#RMSSD – miara pokazująca możliwe zmiany tętna
#pNN50 - odsetek kolejnych odstępów NN różniących się od odstępu poprzedzającego o ponad 50 ms.
#pNN20 – odsetek kolejnych odstępów NN różniących się od odstępu poprzedzającego o ponad 20 ms


indeksy_zdrowych = []
indeksy_chorych = []

for i in tabele_zdrowych:
    x = i.index[i['R_peak'] == 1]
    indeksy_zdrowych.append(x)

for j in tabele_chorych:
    y = j.index[j['R_peak'] == 1]
    indeksy_chorych.append(y)



rr_chorych = []
rr_zdrowych = []

for i in tabele_chorych:
    values = i[i["R_peak"] == 1]["time"].diff()[1:].values.tolist()
    x = np.insert(values, 0, 0, axis = 0) # do listy pochodnych dodajemy 0 na początku listy, tak by zgadzała się ilość danych
    rr_chorych.append(x)

for i in tabele_zdrowych:
    values = i[i["R_peak"] == 1]["time"].diff()[1:].values.tolist()
    x = np.insert(values, 0, 0, axis = 0)
    rr_zdrowych.append(x)


rr_chorych_ilosc = []
rr_zdrowych_ilosc = []

for i in rr_chorych:
    x = len(i)
    rr_chorych_ilosc.append(x)

for i in rr_zdrowych:
    x = len(i)
    rr_zdrowych_ilosc.append(x)

#rr_zdrowych_ilosc = rr_zdrowych.count()

#print(rr_zdrowych)
#print(rr_chorych)



rr_mean_zdr = []
rr_mean_ch = []

for i in(rr_zdrowych):
    x = np.mean(i)
    rr_mean_zdr.append(x)

for i in(rr_chorych):
    x = np.mean(i)
    rr_mean_ch.append(x)

#print(np.mean(rr_mean_zdr))
#print(np.mean(rr_mean_ch))
# Porównująć średnie wartości RR u osób zdrowych i chorych, widać, że nie różnią się tak bardzo od siebie.


SDNN_zdrowych = []
SDNN_chorych = []

for i in rr_zdrowych:
    x = (np.std(i)) * 1000
    SDNN_zdrowych.append(x)

for i in rr_chorych:
    x = (np.std(i)) * 1000
    SDNN_chorych.append(x)



RMSSD_zdrowych = []
RMSSD_chorych = []

for i in rr_zdrowych:
    x = (np.sqrt(np.mean(np.square(np.diff(i))))) * 1000
    RMSSD_zdrowych.append(x)

for i in rr_chorych:
    x = (np.sqrt(np.mean(np.square(np.diff(i))))) * 1000
    RMSSD_chorych.append(x)

#print(RMSSD_zdrowych)
#print(RMSSD_chorych)



pNN50_zdrowych = []
pNN50_chorych = []

for i in rr_zdrowych:
    x = (np.sum((np.abs(np.diff(i)) > 0.05) * 1) / len(i)) * 100
    pNN50_zdrowych.append(x)

for i in rr_chorych:
    x = (np.sum((np.abs(np.diff(i)) > 0.05) * 1) / len(i)) * 100
    pNN50_chorych.append(x)

pNN20_zdrowych = []
pNN20_chorych = []

for i in rr_zdrowych:
    x = (np.sum((np.abs(np.diff(i)) > 0.02) * 1) / len(i)) * 100
    pNN20_zdrowych.append(x)

for i in rr_chorych:
    x = (np.sum((np.abs(np.diff(i)) > 0.02) * 1) / len(i)) * 100
    pNN20_chorych.append(x)


#%% Robimy wizualizację danych. Rusyjemy histogramy oddechów zarówno dla zdrowych, jak i chorych pacjentów.

for i, names in zip(tabele_zdrowych, names_zdrowych):
    i["Resp"].hist(bins = 75)
    plt.tight_layout()
    plt.title("Histogram oddechu pacjenta {}".format(names))
    plt.savefig(os.path.join(KATALOG_WYKRESOW, "Histogram {}.jpg".format(names)), dpi = 300)
    plt.show()


for i, names in zip(tabele_chorych, names_chorych):
    i["Resp"].hist(bins = 75)
    plt.tight_layout()
    plt.title("Histogram oddechu pacjenta {}".format(names))
    plt.savefig(os.path.join(KATALOG_WYKRESOW, "Histogram {}.jpg".format(names)), dpi = 300)
    plt.show()

#%% Liczymy średnie ciśnienie skurczowe

srednie_cisnienie_zdrowych = []
srednie_cisnienie_chorych = []

for i in tabele_zdrowych:
    x = np.mean(i["SBP"])
    srednie_cisnienie_zdrowych.append(x)

for i in tabele_chorych:
    x = np.mean(i["SBP"])
    srednie_cisnienie_chorych.append(x)

print(np.mean(srednie_cisnienie_zdrowych))
print(np.mean(srednie_cisnienie_chorych))

# Można zauważyć, że średnie ciśnienie osób chorych, jest większe, niż u osób zdorwych.

#%% Tworzymy fale wdechów i wydechów pacjentów
# Liczymy pochodne oddechów. Jeżeli są dodatnie to wtedy jest wdech, jeżeli są ujemne, to mamy doczyniania z wydechem.
# Wygładzamy również krzywą oddechową za pomocą metody Moving Average - MA (Średniej kroczącej), o kroku równym 1000, ponieważ
# przy kilkuset lub kilkunastuset tysiącach danych, krok ten jest odpowiednio duży by wygładzić krzywą.

for i in tabele_zdrowych:
    i["MA"] = i["Resp"].rolling(1000).mean()
    values = np.diff(i["MA"])
    x = np.insert(values, 0, 0, axis = 0)
    i["Oddech"] = x    # Tworzymy kolumnę w naszych dataframe'ach, która przedstawia nam jak zmienia się oddech.


for i in tabele_chorych:
    i["MA"] = i["Resp"].rolling(1000).mean()
    values = np.diff(i["MA"])
    x = np.insert(values, 0, 0, axis = 0)
    i["Oddech"] = x

#%% Kolejna wizualizacja danych.
# Robimy wykrech podzielony na oddechy. Kolorem niebieskim na wykresie zaznaczamy wdechy, zielonym wydechy.
# Wygładzamy wykresy

start = 10000
koniec= 30000

for i, names in zip(tabele_zdrowych, names_zdrowych):
    i["B"] = i["R_peak"] * i["MA"]
    i["B"].where(i["B"] > 0, np.nan, inplace = True)
    plt.plot(i['time'][start : koniec], i['MA'].where(i["Oddech"] < 0)[start : koniec],'g.',
            markersize = 4)
    plt.plot(i['time'][start : koniec], i['MA'].where(i["Oddech"] >= 0)[start : koniec],'b.',
            markersize = 4)
    plt.plot(i['time'][start : koniec], i['B'][start : koniec],'rx',
             markersize = 10)
    plt.title("oddech {}".format(names))
    custom_lines = [Line2D([0], [0], color = "blue", lw = 4),
                    Line2D([0], [0], color = "green", lw = 4)]      # Tworzymy legendę
    plt.legend(custom_lines, ["Wdechy", "Wydechy"])
    plt.savefig(os.path.join(KATALOG_WYKRESOW, "Oddech {}.jpg".format(names)), dpi = 300 ) 
    plt.show()

for i, names in zip(tabele_chorych, names_chorych):
    i["B"] = i["R_peak"] * i['MA']
    i["B"].where(i["B"] > 0, np.nan, inplace = True)
    plt.plot(i['time'][start : koniec], i['MA'].where(i["Oddech"] < 0)[start : koniec],'g.',
            markersize = 4)
    plt.plot(i['time'][start : koniec], i['MA'].where(i["Oddech"] >= 0)[start : koniec],'b.',
            markersize = 4)
    plt.plot(i['time'][start : koniec], i['B'][start : koniec],'rx',
             markersize = 10)
    custom_lines = [Line2D([0], [0], color = "blue", lw=4),
                    Line2D([0], [0], color = "green", lw=4)]
    plt.legend(custom_lines, ["Wdechy", "Wydechy"])
    plt.title("oddech {}".format(names))
    plt.savefig(os.path.join(KATALOG_WYKRESOW, "Oddech {}.jpg".format(names)), dpi = 300 ) 
    plt.show()

# Z Obrazków widać, różnicę oddechu osób chorych, a zdrowych. Oddech osób chorych,
# w przeciwieństwie do osób zdrowych, jest częściej nieregularny, płytszy. Osoby zdrowsze zazwyczaj biorą więcej
# wdechów i wydechów w danym przedziale czasowym. U osób chorych, jak i zdrowych, można zauważyć, że występują pewne załamania
# oddechowe, gdzie pacjent podczas wydechu lub wdechu, bierze bardzo krótki, znacznie mniejszy oddech.

#%% Liczymy ilość R_peaków podczas wdechów i wydechów


for i in tabele_zdrowych:
    values = np.diff(i["Resp"])
    x = np.insert(values, 0, 0, axis = 0)
    i["Oddech"] = x


for i in tabele_chorych:
    values = np.diff(i["Resp"])
    x = np.insert(values, 0, 0, axis = 0)
    i["Oddech"] = x

ilość_R_peak_wdechów_zdrowych = []
ilość_R_peak_wydechów_zdrowych = []
ilość_R_peak_wdechów_chorych = []
ilość_R_peak_wydechów_chorych = []

# Jeżeli R_peak występuje tzn i["R_peak"] == 1, oraz pochodna oddechu jest dodatnia, to wtedy jest to R_peak przy wdechu.
# Jeżeli zaś pochodna oddechu jest ujemna, to wtedy jest to R-peak przy wydechu.

for i in tabele_zdrowych:
    x = np.sum((i["R_peak"] == 1) & (i["Oddech"] > 0))
    ilość_R_peak_wdechów_zdrowych.append(x)

for i in tabele_zdrowych:
    x = np.sum((i["R_peak"] == 1) & (i["Oddech"] < 0))
    ilość_R_peak_wydechów_zdrowych.append(x)

for i in tabele_chorych:
    x = np.sum((i["R_peak"] == 1) & (i["Oddech"] > 0))
    ilość_R_peak_wdechów_chorych.append(x)

for i in tabele_chorych:
    x = np.sum((i["R_peak"] == 1) & (i["Oddech"] < 0))
    ilość_R_peak_wydechów_chorych.append(x)

#%% Wzrosty a i Spadki d rytmu serca przy wdechach i wydechach

# Liczymy przyspieszenia i zwolnienia rytmu serca. Jest to pochodna z RR.
Przys_i_zwol_zdr = []

for i in rr_zdrowych:
    values = np.diff(i).tolist()
    x = np.insert(values, 0, 0, axis = 0)
    Przys_i_zwol_zdr.append(x)

# Zmieniamy wartości z numpy arraya na wartości w liście.

Przys_zwol_zdr = []
for i in Przys_i_zwol_zdr:
    x = i.tolist()
    Przys_zwol_zdr.append(x)


Przys_i_zwol_ch = []
for i in rr_chorych:
    x = np.diff(i).tolist()
    x = np.insert(values, 0, 0, axis = 0)
    Przys_i_zwol_ch.append(x)

Przys_zwol_ch = []
for i in Przys_i_zwol_ch:
    x = i.tolist()
    Przys_zwol_ch.append(x)


# Bierzemy indeksy R_peaków

indeksy_R_peak_zdrowych = []

for i in tabele_zdrowych:
    x = i.index[i["R_peak"] == 1].tolist()
    indeksy_R_peak_zdrowych.append(x)

# Teraz sprawdzamy jakie są wartości oddechowe dla tych R_peaków

Oddech_zdrowych = []

for i in tabele_zdrowych:
    x = i.query("R_peak == 1")["Oddech"].tolist()
    Oddech_zdrowych.append(x)

a_d_zdr = []

# Indeksów R_peak jak i ilości przyspieszeń i zwolnień rytmu serca jest identyczna ilość. Ma to sens, ponieważ
# przyspieszenia i zwolnienia rytmu serca to pochodna z RR, a RR to czas między kolejnymi R_peakami.

# Tworzymy teraz listy zawierające indeks R_peaku, jaka jest dla niego wartość oddechowa oraz wartość przyspieszenia lub
# zwolnienia. Jeżeli wartość ta jest dodatnia to mamy przyspieszenie, w przeciwnym wypadku mamy zwolnienie.

for i in range(len(indeksy_R_peak_zdrowych)):
    idk = list(map(list,zip(indeksy_R_peak_zdrowych[i], Oddech_zdrowych[i], Przys_zwol_zdr[i])))
    a_d_zdr.append(idk)


indeksy_R_peak_chorych = []

for i in tabele_chorych:
    x = i.index[i["R_peak"] == 1].tolist()
    indeksy_R_peak_chorych.append(x)

Oddech_chorych = []

for i in tabele_chorych:
    x = i.query("R_peak == 1")["Oddech"].tolist()
    Oddech_chorych.append(x)

a_d_ch = []

for i in range(len(indeksy_R_peak_chorych)):
    idk = list(map(list,zip(indeksy_R_peak_chorych[i], Oddech_chorych[i], Przys_zwol_ch[i])))
    a_d_ch.append(idk)


# Dzielimy nasze dane dla kazdego pacjenta i sprawdzamy kiedy występują przyspieszenia i zwolnienia przy wdechach i wydechach
# Jeżeli drugi element z listy (z indeksem [1]), czyli oddech, jest ujemny to mamy wydech, jeżeli jest dodatni to mamy wdech.
# Jeżeli trzeci element z listy (z indeksem [2]), czyli przyspieszenie lub zwolneinie jest ujemny to mamy zwolnienie,
# jeżeli jest dodatni to mamy przyspieszenie.
# Z tymi informacjami wrzucamy odpowiednie listy z danymi do odpowiednich im kategorii, czyli czy było przyspieszenie/zwolnienie
# na wdechu/wydechu.

a_wdech_zdr = []
a_wydech_zdr = []
d_wdech_zdr = []
d_wydech_zdr = []
a_wdech_ch = []
a_wydech_ch = []
d_wdech_ch = []
d_wydech_ch = []


for i in a_d_zdr:
    x = [j for j in i if j[1] > 0 and j[2] > 0]
    a_wdech_zdr.append(x)


for i in a_d_zdr:
    x = [j for j in i if j[1] < 0 and j[2] > 0]
    a_wydech_zdr.append(x)

for i in a_d_zdr:
    x = [j for j in i if j[1] > 0 and j[2] < 0]
    d_wdech_zdr.append(x)

for i in a_d_zdr:
    x = [j for j in i if j[1] < 0 and j[2] < 0]
    d_wydech_zdr.append(x)



for i in a_d_ch:
    x = [j for j in i if j[1] > 0 and j[2] > 0]
    a_wdech_ch.append(x)


for i in a_d_ch:
    x = [j for j in i if j[1] < 0 and j[2] > 0]
    a_wydech_ch.append(x)

for i in a_d_ch:
    x = [j for j in i if j[1] > 0 and j[2] < 0]
    d_wdech_ch.append(x)

for i in a_d_ch:
    x = [j for j in i if j[1] < 0 and j[2] < 0]
    d_wydech_ch.append(x)

#%% Robimy listy ilosci zwolnień i przyspieszeń rytmu serca przy wdechach i oddechach

a_wdech_zdr_ilosc = []
a_wydech_zdr_ilosc = []
d_wdech_zdr_ilosc = []
d_wydech_zdr_ilosc = []
a_wdech_ch_ilosc = []
a_wydech_ch_ilosc = []
d_wdech_ch_ilosc = []
d_wydech_ch_ilosc = []

for i in a_wdech_zdr:
    x = len(i)
    a_wdech_zdr_ilosc.append(x)

for i in a_wydech_zdr:
    x = len(i)
    a_wydech_zdr_ilosc.append(x)

for i in d_wdech_zdr:
    x = len(i)
    d_wdech_zdr_ilosc.append(x)

for i in d_wydech_zdr:
    x = len(i)
    d_wydech_zdr_ilosc.append(x)

for i in a_wdech_ch:
    x = len(i)
    a_wdech_ch_ilosc.append(x)

for i in a_wydech_ch:
    x = len(i)
    a_wydech_ch_ilosc.append(x)

for i in d_wdech_ch:
    x = len(i)
    d_wdech_ch_ilosc.append(x)

for i in d_wydech_ch:
    x = len(i)
    d_wydech_ch_ilosc.append(x)

        
#%% wzrosty i, spdaki v cisnienia skurczowego

for i in tabele_zdrowych:
    values = np.diff(i["SBP"])
    x = np.insert(values, 0, 0, axis = 0)
    i["SBP diff"] = x

# Tutaj robimy analogicznie jak w poprzednich metodach. Wzrosty i spadki ciśnienia skurczowego to pochodna z "SBP".
# Jeżeli pochodna ta jest większa od 0 to mamy doczynienia ze wzrostem, jeżeli jest mniejsza od 0 to jest spadek.
# Teraz grupujemy dane na podstawie tego, znaku oddechu i pochodnej z SBP. Przy odpowiednich znakach, dane wrzucą się do
# odpowiednich list. Liczymy także ilość elementów w listach.

i_wdech_zdr = []
i_wydech_zdr = []
v_wdech_zdr = []
v_wydech_zdr = []


for i in tabele_zdrowych:
    x = i["SBP diff"][(i["SBP diff"] > 0) & (i["Oddech"] > 0)].count()
    i_wdech_zdr.append(x)


for i in tabele_zdrowych:
    x = i["SBP diff"][(i["SBP diff"] > 0) & (i["Oddech"] < 0)].count()
    i_wydech_zdr.append(x)


for i in tabele_zdrowych:
    x = i["SBP diff"][(i["SBP diff"] < 0) & (i["Oddech"] > 0)].count()
    v_wdech_zdr.append(x)


for i in tabele_zdrowych:
    x = i["SBP diff"][(i["SBP diff"] < 0) & (i["Oddech"] < 0)].count()
    v_wydech_zdr.append(x)




for i in tabele_chorych:
    values = np.diff(i["SBP"])
    x = np.insert(values, 0, 0, axis = 0)
    i["SBP diff"] = x


i_wdech_ch = []
i_wydech_ch = []
v_wdech_ch = []
v_wydech_ch = []


for i in tabele_chorych:
    x = i["SBP diff"][(i["SBP diff"] > 0) & (i["Oddech"] > 0)].count()
    i_wdech_ch.append(x)


for i in tabele_chorych:
    x = i["SBP diff"][(i["SBP diff"] > 0) & (i["Oddech"] < 0)].count()
    i_wydech_ch.append(x)


for i in tabele_chorych:
    x = i["SBP diff"][(i["SBP diff"] < 0) & (i["Oddech"] > 0)].count()
    v_wdech_ch.append(x)


for i in tabele_chorych:
    x = i["SBP diff"][(i["SBP diff"] < 0) & (i["Oddech"] < 0)].count()
    v_wydech_ch.append(x)


#%% Informacje o pacjentach

# Tworzymy listę informacji o pacjentach

inf_zdrowe_osoby = []



for i in range(len(tabele_zdrowych)):
    zdrowe_osoby = [names_zdrowych[i], SDNN_zdrowych[i], RMSSD_zdrowych[i], pNN50_zdrowych[i],
                pNN20_zdrowych[i], srednie_cisnienie_zdrowych[i],
                ilość_R_peak_wdechów_zdrowych[i], ilość_R_peak_wydechów_zdrowych[i],
                a_wdech_zdr_ilosc[i], d_wdech_zdr_ilosc[i], a_wydech_zdr_ilosc[i], d_wydech_zdr_ilosc[i],
                i_wdech_zdr[i], i_wydech_zdr[i], v_wdech_zdr[i], v_wydech_zdr[i]]
    inf_zdrowe_osoby.append(zdrowe_osoby)

# Tworzymy dataframe'y z informacjami o pacjentach

dataframes_zdrowych = []
for i in range(len(tabele_zdrowych)):
    x = pd.DataFrame(inf_zdrowe_osoby[i]).transpose()
    x.columns = ["Pacjent", "SDNN", "RMSSD", "pNN50", "pNN20",
                 "Średnie ciśnienie skurczowe",
                 "Ilość R-peaków przy wdechu",
                 "Ilość R-peaków przy wydechu",
                 "Ilość przyspieszeń rytmu serca przy wdechu",
                 "Ilość zwolnień rytmu serca przy wdechu",
                 "Ilość przyspieszeń rytmu serca przy wydechu",
                 "Ilość zwolnień rytmu serca przy wydechu",
                 "Ilość wzrostów ciśnienia skurczowego przy wdechu",
                 "Ilość wzrostów ciśnienia skurczowego przy wydechu",
                 "Ilość spadków ciśnienia skurczowego przy wdechu",
                 "Ilość spadków ciśnienia skurczowego przy wydechu"]
    dataframes_zdrowych.append(x)



inf_chore_osoby = []
for i in range(len(tabele_chorych)):
    chore_osoby = [names_chorych[i], SDNN_chorych[i], RMSSD_chorych[i], pNN50_chorych[i],
                pNN20_chorych[i], srednie_cisnienie_chorych[i],
                ilość_R_peak_wdechów_chorych[i], ilość_R_peak_wydechów_chorych[i],
                a_wdech_ch_ilosc[i], d_wdech_ch_ilosc[i], a_wydech_ch_ilosc[i], d_wydech_ch_ilosc[i],
                i_wdech_ch[i], i_wydech_ch[i], v_wdech_ch[i], v_wydech_ch[i]]
    inf_chore_osoby.append(chore_osoby)


dataframes_chorych = []
for i in range(len(tabele_chorych)):
    x = pd.DataFrame(inf_chore_osoby[i]).transpose()
    x.columns = ["Pacjent", "SDNN", "RMSSD", "pNN50", "pNN20",
                 "Średnie ciśnienie skurczowe",
                 "Ilość R-peaków przy wdechu",
                 "Ilość R-peaków przy wydechu",
                 "Ilość przyspieszeń rytmu serca przy wdechu",
                 "Ilość zwolnień rytmu serca przy wdechu",
                 "Ilość przyspieszeń rytmu serca przy wydechu",
                 "Ilość zwolnień rytmu serca przy wydechu",
                 "Ilość wzrostów ciśnienia skurczowego przy wdechu",
                 "Ilość wzrostów ciśnienia skurczowego przy wydechu",
                 "Ilość spadków ciśnienia skurczowego przy wdechu",
                 "Ilość spadków ciśnienia skurczowego przy wydechu"]
    dataframes_chorych.append(x)

#%% Tworzymy ostateczne tabele z wynikami, gdzie wszyscy pacjenci zdrowi są w jednej tabeli, tak samo jak pacjenci chorzy.

tabela_zdrowych = pd.concat([df.set_index('Pacjent') for df in dataframes_zdrowych])
tabela_chorych = pd.concat([df.set_index('Pacjent') for df in dataframes_chorych])

tabela_pacjentów = pd.concat([tabela_zdrowych, tabela_chorych])

#%%
#                                                  PROJEKT 2

#%% Standaryzacja danych
tabela_pacjentów_test = (tabela_pacjentów - tabela_pacjentów.mean()) / tabela_pacjentów.std()
print(tabela_pacjentów_test)

#%% Dzielimy na chorych i zdrowych, gdzie 1 oznacza zdrowych, a 2 oznacza chorych

tabela_pacjentów_test["Status Pacjenta"] = np.nan

for i in tabela_pacjentów_test.index:
    if i[4] == "1":
        tabela_pacjentów_test['Status Pacjenta'][i] = 1
    if i[4] == "2":
        tabela_pacjentów_test["Status Pacjenta"][i] = 2


#%% Tworzymy wykres PCA

features = ["SDNN", "RMSSD", "pNN50", "pNN20",
             "Średnie ciśnienie skurczowe",
             "Ilość R-peaków przy wdechu",
             "Ilość R-peaków przy wydechu",
             "Ilość przyspieszeń rytmu serca przy wdechu",
             "Ilość zwolnień rytmu serca przy wdechu",
             "Ilość przyspieszeń rytmu serca przy wydechu",
             "Ilość zwolnień rytmu serca przy wydechu",
             "Ilość wzrostów ciśnienia skurczowego przy wdechu",
             "Ilość wzrostów ciśnienia skurczowego przy wydechu",
             "Ilość spadków ciśnienia skurczowego przy wdechu",
             "Ilość spadków ciśnienia skurczowego przy wydechu"]
x = tabela_pacjentów_test.loc[:, features].values
y = tabela_pacjentów_test.loc[:,['Status Pacjenta']].values

pca = PCA(n_components = 2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

#%%

principalDf["Status Pacjenta"] = tabela_pacjentów_test["Status Pacjenta"].tolist()

finalDf = principalDf
#%%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = [1, 2]
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Status Pacjenta'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

print(pca.explained_variance_ratio_)

#%% Standardowe oznaczenia przetwarzanych danych i ogólny ogląd

X , y  = tabela_pacjentów, tabela_pacjentów_test["Status Pacjenta"]
print('dane do klasyfikacji:\n',X,'\nrozmiar danych:', X.shape)
print('znane etykiety:\n', y ,'\nrozmiar etykiety', y.shape)
y_2 = (y == 2)
print('binarne etykiety:\n', y_2 ,'\nrozmiar etykiety', y_2.shape)

#%% TRAIN and TEST sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print('X train', X_train.shape, 'X test', X_test.shape)
print('y train', y_train.shape, 'y test', y_test.shape)

y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

#%%% STARTOWA PRÓBKA  klasyfikacji: cały zbiór trenujący, 
#                                  algorytm przy domylnych ustawieniach

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter = 50, tol = None,
                       loss = "log", penalty = "l2")
# linear discriminant classifier
#loss="hinge": (soft-margin) linear Support Vector Machine,
#loss="modified_huber": smoothed hinge loss,
#loss="log": logistic regression,

sgd_clf.fit(X_train, y_train_2)

print('współczynniki regresji liniowej (aby poczuć liniowe rozwiązanie) ')
print(sgd_clf.intercept_, '+' , sgd_clf.coef_) 

#%%  ZACZYNAMY WALIDACJE

from sklearn.model_selection import cross_val_score

moje_cv = 5
napis= 'dokladnosć: ilosc dobrych predykcji'
dokladnosc_klasyfikacji = cross_val_score(sgd_clf, X_train,  y_train_2, 
                                              cv = moje_cv, scoring="accuracy")
print(napis, dokladnosc_klasyfikacji)

napis= 'prezycja : TP/(TP+NP) ilosc dobrych predykcji w klasie pozytwnej (nie 2)'
precyzja_klasyfikacji = cross_val_score(sgd_clf, X_train,  y_train_2, 
                                              cv=moje_cv, scoring="precision")
print(napis, precyzja_klasyfikacji)


napis= 'recall : TP(TP+FN)  ile instancji klasy pozytwnej (nie 2) zostalo dobrze rozpoznane'
czulosc_klasyfikacji = cross_val_score(sgd_clf, X_train,  y_train_2, 
                                              cv=moje_cv, scoring="recall")

print(napis, czulosc_klasyfikacji)

#%% WSTEPNE PODSUMOWANIE

print("srednia i std dokładnosci %.3f"% dokladnosc_klasyfikacji.mean(), 
      " +/- %.3f"% dokladnosc_klasyfikacji.std())

print("srednia i std    precyzji  %.3f"% precyzja_klasyfikacji.mean() ,
      " +/- %.3f"%precyzja_klasyfikacji.std()  )


print("srednia i std    czułosci %.3f"%czulosc_klasyfikacji.mean(),
      " +'- %.3f"%czulosc_klasyfikacji.std())

#%% Konstrukcja tablica pomylek ( macierz błędow   )

from sklearn.model_selection import cross_val_predict

y_train_predict = cross_val_predict(sgd_clf, X_train, y_train_2, cv= moje_cv)
#  predykcja jest uzyskana z walidacji krzyżowej, czyli na zbiorach walidacyjnych

#%%
from sklearn.metrics import confusion_matrix

confusion =  confusion_matrix(y_train_2, y_train_predict)

precision = (confusion[1,1]/(confusion[1,1] + confusion[0,1]))
recall = (confusion[1,1]/(confusion[1,1] + confusion[1,0]))

print('confusion matrix: \n', confusion, '\n')
print( 'TN=', confusion[0,0], '\tFP=', confusion[0,1])
print( 'FN=', confusion[1,0], '\tTP=', confusion[1,1])
print('\nprecision=  %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall= %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])))

from sklearn.metrics import precision_score, recall_score
print('\n 2-ki wsrod sklasyfikowanych jako 2 to %.2f'% precision_score(y_train_2, y_train_predict))
print('\n 2-ki rozpoznano w %.2f przypadkach  '% recall_score(y_train_2, y_train_predict))

#%%  KOMPROMIS
#użycie metody decision_function() glownego obiektu
y_scores = cross_val_predict(sgd_clf, X_train, y_train_2, cv = 5,
                             method="decision_function")
                        # domyslnie jest method='predict'
               
#%% https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision-recall

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_2, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-200000, 100000])
plt.savefig("precision_recall_vs_threshold_plot",dpi=300)
plt.show()
#%%

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-20000, 20000])
plt.show()

#%%
plt.figure(figsize=(8, 4))
plt.plot(recalls, precisions )
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Recall vs Precision')
plt.show


threshold_90_precision = thresholds[np.argmax(precisions >= 0.75)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
print(precision_score(y_train_2, y_train_pred_90))
print(recall_score(y_train_2, y_train_pred_90))

#%% WYSZUKIWANIE NAKLEPSZYCH WARTOŚCI HIPERPARAMETROW popularna metoda

from sklearn.model_selection import GridSearchCV

params = {
    'loss' : ['hinge', 'log', 'squared_hinge'] ,#'modified_huber', 'perceptron'],
    'alpha' : [ 0.01, 0.1],
    'penalty' :['l2','l1']#,'elasticnet','none']
        }

sgd_clf= SGDClassifier(max_iter =100)
grid = GridSearchCV ( sgd_clf, param_grid =params, cv=moje_cv, scoring ='f1',
                     return_train_score=True)

grid.fit(X_train, y_train_2)

cv_res = grid.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(mean_score, params)
    

#%% Ostateczne przetwarzanie
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

final_model = grid.best_estimator_


final_model_param = final_model.get_params()
final_training = final_model.fit(X_train, y_train_2)

final_predictions = final_training.predict(X_test)

#final_confusion =  confusion_matrix(final_training, final_predictions)

final_f1 = f1_score(y_test_2, final_predictions)
final_accuracy = accuracy_score(y_test_2, final_predictions)
final_recall = recall_score(y_test_2, final_predictions)
final_precision = precision_score(y_test_2, final_predictions)

print('f1 najlepszego rozwiazania SGDClassifier ',"%.2f"%final_f1)
print('Dokładność najlepszego rozwiązania SGDClassifier: ', "%.2f" % final_accuracy)
print('recall najlepszego rozwiazania SGDClassifier ',"%.2f"%final_recall)
print('presision najlepszego rozwiazania SGDClassifier ',"%.2f"%final_precision)

#%% KRZYWA UCZENIA

from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    final_model, X_train, y_train_2, cv=moje_cv, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="f1", n_jobs=-1)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'b-o', label="Training F1 Score")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'g-o', label="Validation F1 Score")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.show()



