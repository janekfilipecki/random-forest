
 Uczenie Maszynowe - Projekt semestralny - Drzewa decyzyjne i lasy losowe
=============



## Konfiguracja środowiska

W projekcie korzystamy z Pipenv, który jest narzędziem do zarządzania zależnościami w projekcie Pythona, łączące w sobie funkcje pip (do instalacji pakietów) i virtualenv (do izolowania środowiska).

1. **Instalacja Pipenv**: Upewnij się, że masz zainstalowany Python w wersji 3.12, lub `pyenv`, którego pipenv użyje do zainstalowania odpowiedniej wersji pythona. Następnie zainstaluj Pipenv za pomocą polecenia pip:

   ```bash
   pip install pipenv
   ```

2. **Inicjacja projektu**: Przejdź do głównego katalogu swojego projektu w terminalu i uruchom:

   ```bash
   pipenv install
   ```

   Komenda ta z wykorzystaniem `Pipfile` stworzy odpowiednie środowisko wirtualne do pracy z projektem.

3. **Aktywacja środowiska wirtualnego**: Aby aktywować środowisko wirtualne stworzone przez Pipenv, użyj:

   ```bash
   pipenv shell
   ```

4. **Uruchamianie skryptów**: Aby uruchomić skrypty w środowisku Pipenv, użyj:

   ```bash
   pipenv run nazwa_skryptu.py
   ```

   lub jeśli aktywne jest środowisko po wcześniejszeym użyciu `pipenv shell`, po prostu

   ```bash
   python nazwa_skryptu.py

   ```

5. **Dezaktywacja środowiska wirtualnego**: Aby wyjść z wirtualnego środowiska, użyj komendy:

   ```bash
   exit
   ```

6. **Instalacja pakietów**: W celu instalacji pakietów i tym samym dodawania ich do paczki użyj komendy

   ```bash
   pipenv install nazwa_pakietu
   ```

   Wykonanie tej komendy zmieni Pipfile i Pipfile.lock. Możesz również wprowadzać ręczne zmiany do Pipfile, lecz pamiętaj, by każdorazowo po takowej zmianie uruchamiać komendę `pipenv install` by wygenerować `Pipfile.lock` i stworzyć odpowiednie, aktualne środowisko.

Pamiętaj, że plik `Pipfile.lock` automatycznie zapisuje dokładne wersje zainstalowanych pakietów, aby zapewnić spójność środowiska na różnych maszynach. Przy kolejnych uruchomieniach projektu, zaleca się używanie poleceń `pipenv install` w celu zainstalowania zależności zdefiniowanych w pliku `Pipfile`.

## Uruchamianie testów

Upewnij się, że poprawnie wykonałeś konfigurację środowiska. Następnie uruchom środowisko i wpisz następującą komendę, znajdując się w folderze korzenia projektu: 

   ```bash
   pytest tests/
   ```


Opis projektu
=============

Celem projektu jest zapoznanie się z algorytmem ID3 oraz Lasem Losowym
poprzez wykonanie naszej własnej implementacji algorytmów, a następnie
porównanie naszej implementacji z implementacją z biblioteki
scikit-learn. Obydwie implementacje zostaną wykorzystane do stworzenia
modelu dla wybranych problemów klasyfikacji, na odpowiednio
przygotowanych zbiorach danych przedstawionych w sekcji. Następnie
porównamy implementacje na bazie wybranych metryk, takich jak precision,
recall, accuracy oraz F1.

Opis algorytmów
===============

ID3
---

ID3 (Iterative Dichotomiser 3) to algorytm używany do generowania drzew
decyzyjnych. Opiera się on na obliczaniu dwóch parametrów: Entropii -
będącej miarą zróżnicowania danych oraz wynikającego z niej Przyrostu
Wiedzy (IG - Information Gain).

**Entropia**: $$H(S) = - \sum_{c \in C} p(c)\log_2 p(c)$$ Gdzie:

$S$ - Aktualny zbiór, dla którego liczymy entropię

$C$ - Zbiór klas w zbiorze $S$

$p(c)$ - Prawdopodobieństwo wystąpienia klasy $c$ w zbiorze $S$

Entropia zbioru jest tym mniejsza, im większa jest dysproporcja klas w
danym zbiorze.

**Przyrost Wiedzy**:
$$IG(A) = H(S) - H(S|A)= H(S) - \sum_{v \in V(A)} \frac{|S_v|}{S}H(S_v)$$
Gdzie:

$S$ - Aktualny zbiór danych, który dzielić będziemy za pomocą wartości
wybieranego atrybutu

$A$ - Wybierany atrybut, według którego dzielić będziemy zbiór danych

$V(A)$ - Zbiór wartości dopuszczalnych atrybutu

$S_v$ - Podzbiór danych, w którym próbki przyjmują wartości $v$ atrybutu
$A$

Przyrost informacyjny jest największy, kiedy suma entropii podzbiorów
jest najmniejsza, czyli podzbiory posiadają w sobie duże dysproporcje
klas. Ergo dzielą zbiór w taki sposób, że podzbiory jak najlepiej
reprezentują jedną z klas docelowych i mają jak najmniej nieczystości.

Algorytm iteracyjnie tworzy węzły drzewa, kolejno wybierając atrybuty
tak, by przyrost informacyjny po danym podziale był największy.
Minimalizując entropię warunkową dla danego podziału, czyli
maksymalizując redukcję nieczystości w wyniku podziału, algorytm tworzy
drzewo tak, by jak najlepiej zdekomponować zbiór treningowy. Działanie
algorytmu ID3 można opisać w następujący sposób:

    1. Oblicz IG dla każdego atrybutu
    2. Wybierz atrybut z najwyższym IG
    3. Podziel zbiór przykładów uczących ze względu na wartości atrybutu A na rozłączne podzbiory 
    4. Dodaj do drzewa krawędzie z warunkami:
       dla każdego v należącego do V(A), dodajemy warunek V(A) = v (tworzymy poddrzewo)
    5. Dla każdego poddrzewa wykonaj kroki od 1.
    6. W każdej iteracji jeden atrybut jest usuwany. Algorytm zatrzymuje się, 
       gdy do rozpatrzenia nie pozostanie juz żaden atrybut lub wszystkie przykłady 
       w danym podrzewie mają tą samą wartość atrybutu decyzyjnego.

Pseudokod algorytmu ID3

W zależności od implementacji algorytmu ID3, rozgałęzienia, mogą dzielić
drzewo binarnie lub na więcej węzłów. Warunki podziału mogą być
zdefiniowane dowolnie: na podstawie wartości atrybutów, na podstawie
podzbiorów wartości atrybutu lub na podstawie przedziałów wartości
atrybutu, w przypadkach ciągłych. W klasycznej implementacji algorytmu
nie posiadał żadnego z tych rozszerzeń. Nie posiadał on również
przycinania (pruning), w związku z czym jednym z głównych problemów
algorytmu ID3, jest tworzenie modeli o niskim obciążeniu i wysokiej
wariancji, czyli nadmiernie się dopasowujących. Jednak kolejne iteracje
algorytmu ID3 takie jak C4.5 lub C5 zawierają właśnie te poprawki.

Warto w tym miejscu również wspomnieć o pokrewnym algorytmie, jakim jest
CART (Classification and Regression Trees), który zamiast entropii
warunkowej wykorzystuje Warunkowy indeks Giniego. Oprócz tego CART
może wielokrotnie wybierać ten sam atrybut do podziału na różnych
poziomach drzewa, co pozwala na bardziej złożone struktury.

W zależności od możliwości czasowych będziemy chcieli zaimplementować
wersję algorytmu, która będzie w stanie obsłużyć atrybuty kategoryczne
jak i ciągłe oraz będzie posiadać opcję przycinania drzewa, czyli de
facto będzie swoistą wersją algorytmu C4.5.

CART
---

Algorytm CART jest bardzo podobny w działaniu do algorytmów opisanych wyżej. Podstawową różnicą w działaniu jest to, że w algorytmie CART jako kryterium wyboru atrybutu stosowany jest
współczynnik Giniego. Najlepszy podział cechuje się najmniejszą
wartością miary różnorodności, czyli sytuacją, w której po podziale
zróżnicowanie klas w podzbiorach było jak najmniejsze (współczynnik
Giniego przyjmuje najmniejszą wartość).

Indeks Giniego opisuje prawdopodobieństwo wybrania niepoprawnej klasy po wybraniu losowego elementu ze zbioru, ma wobec tego minimalną wartość (najwyższy poziom czystości) 0, oraz maksymalną 0,5. Jeśli Indeks Giniego wynosi 0,5, oznacza to losowy przydział klas. Entropia z kolei jest miarą niepewności i losowości w zbiorze, jest ona miarą logarytmiczną, a indeks Giniego jest miarą liniową.

Efektem tych różnic są różne niuanse sprawiające że miary te zachowują się inaczej w zastosowaniu dla drzew decyzyjnych. Indeks Giniego jest bardziej czuły na rozkład klas, a entropia na ich ilość. Obliczanie indeksu Giniego ma także mniejszą złożoność obliczeniową, przez jego liniowy charakter. Entropia uważana jest jednak za mniej czułą miarą, w porównianiu do IG. Co warto również zaznaczyć, Ig częściej wybiera podziały, których rezultatem są zbiory o zbalansowanych ilościach klas, podczas gdy Entropia częściej wybiera podziały, dla których największa jest redukcja zaszumienia.

Nie ma jednego wygranego w tej bitwie, oba sposoby obliczania odpowiednich podziałów są dobre i znajdują swoje zastosowania. Przejdźmy jednak dalej do opisu Indeksu Giniego.


**Indeks Giniego**

$$I_G(A) = 1 - \sum_{k=1}^{K}p_k^2$$ 
$$p_k = \frac{N_{A,k}}{N_A}$$
Gdzie:

$N_{A,k}$ - Liczba przykładów w zbiorze $A$ z klasą $k$

$N_A$ - Liczba wszystkich przykłady ze zbioru $A$


Na podstawie właśnie tej miary CART znajduje odpowiednie podziały, poniżej znajduje się pseudokod algorytmu CART, który jest bardzo podobny do ID3, jednak zawiera kilka kluczowych różnic.


    1. Znajdź możliwe podziały dla każdego atrybutu
    2. Dla tych par atrybutów i podziałów oblicz Ig
    3. Wybierz najlepszą parę
    4. Na podstawie wybranych podziałów stwórz nowe węzły
    5. Dla potomnych węzłów usuń wybrany poprzednio atrybut 
    6. Dla każdego poddrzewa wykonaj kroki od 1.
    7. W każdej iteracji jeden atrybut jest usuwany. Algorytm zatrzymuje się, 
       gdy do rozpatrzenia nie pozostanie juz żaden atrybut, wszystkie przykłady 
       w danym podrzewie mają tą samą wartość atrybutu decyzyjnego, osiągnięto maksymalną zadaną głębokość drzewa lub ilość przykładów w podzbiorach po podziale byłaby zbyt mała.

Pseudokod algorytmu CART



Las losowy
----------

Las losowy (Random Forest, RF) to algorytm, który wykorzystuje
informacje pozyskane z wielu drzew decyzyjnych. Jest to metoda
uśredniania wyników wielu modeli trenowanych na różnych częściach zbioru
treningowego, której celem jest zmniejszenie wariancji modelu. Jest to
metoda wywodząca się z modelowania zespołowego (Ensemble Learning), a
dokładniej Baggingu, który polega na równoległym trenowaniu wielu
słabych modeli. Druga z dwóch najbardziej prominentnych metod
modelowania zespołowego, czyli Boosting, opiera się na trenowaniu
słabych modeli sekwencyjnie. Wówczas modele kolejno poprawiają swoją
jakość, dopełniając się wzajemnie.

**Bagging** By powiedzieć coś o trenowaniu lasów losowych, najpierw
wprowadzić trzeba pojęcie Baggingu. Jak wyżej wymieniono, jest to metoda
polegająca na trenowaniu równolegle wielu słabych modeli. Metoda oparta
jest na agregacji Bootstrap. Bootstrapping jest statystyczną metodą
szacowania rozkładów błędów estymacji, za pomocą wielokrotnego losowania
ze zwracaniem z próby, jednak jej założenia mają wiele zastosowań w
pokrewnych zadaniach. W tym przypadku próbami bootstrap nazywać będziemy
$k$ podzbiorów próbek ze zbioru trenującego $T$ o rozmiarze $n$,
generowanych za pomocą losowania próbek z rozkładu jednostajnego ze
zwracaniem. Próbkowanie ze zwracaniem zapewnia, że każda próba jest
niezależna od pozostałych.

Bagging zwykle przebiega następująco:

    1. Dla zbioru trenującego T wygeneruj k podzbiorów próbek za pomocą bootstrappingu
    2. Dla każdego podzbioru Ti stwórz z jego pomocą model
    3. Następnie dokonaj predykcji poprzez uśrednienie lub głosowanie większościowe 
       wyników ze wszystkich modeli.

Pseudokod Baggingu

Procedura próbkowania bootstrap pozwala na znaczne zmniejszenie
wariancji modelu, bez istotnego zwiększania jego obciążenia. Dzieje się
to dlatego, że w przeciwieństwie do pojedynczych drzew, które są bardzo
dopasowane i posiadają wysoką wariancję, średnia wyników zbioru takich
modeli posiada dużo mniejszą wariancję, jeśli tylko drzewa nie są ze
sobą skorelowane, co w praktyce jest trudne do zapewnienia. Próbkowanie
bootstrap ma na celu zmniejszenie tych korelacji poprzez losowe
generowanie próbek.

Liczba słabych modeli czy też podzbiorów trenujących $k$ jest
hiperparametrem. W praktyce zazwyczaj ustawia się ją w granicach od
kilkuset do kilku tysięcy słabych uczniów.

**Random Subspace Method** Metoda losowej podprzestrzeni zwana również
Attribute Bagging (lub Feature Bagging) to metoda tworzenia modeli
poprzez trenowanie ich na losowych podzbiorach atrybutów zamiast całego
zbioru atrybutów. Metoda ta ma na celu dalsze zmniejszenie korelacji
pomiędzy modelami.

Attribute Bagging jest bardzo podobną metodą do zwykłego Baggingu, z tym
że w przypadku tej pierwszej próbkować będziemy atrybuty, czyli tak jak
wcześniej opisane, losować ze zwracaniem ze zbioru wszystkich atrybutów.
Wytrenowanie modeli na tak stworzonych podzbiorach, mówiąc żargonem
inżynierskim, pozwala modelom nie skupiać się na atrybutach bardzo
wysoko informatywnych lub/i predyktywnych i tym samym nauczyć się
zależności dużo bardziej wysublimowanych, ciężkich do wyłapania dla
innych metod.

**Trening lasu losowego** Teraz gdy wiemy jak działa zwykły Bagging oraz
Feature Bagging, możemy przejść do opisu algorytmu uczenia lasu
losowego. Las losowy wykorzytuje obie te metody, by jak najbardziej
zmniejszyć korelację pomiędzy modelami.

W przypadku lasu losowego, jak wskazuje logika, modelami są drzewa
decyzyjne, budowane korzystając z jednego z algorytmów drzew
decyzyjnych. Zazwyczaj jest to CART, ze względu na swoją uniwersalność
względem typów atrybutów.

    1. Dla zbioru trenującego T wygeneruj podzbiór n próbek z rozkładu jednostajnego
        1*. Pozostałe |T| - n próbek zapisz jako zbiór weryfiacyjny out-of-bag (oob)
    2. Dla każdego podzbioru Ti wygeneruj podzbiór m wszystkich atrybutów A
    3. Zbuduj drzewo korzystając ze zbioru atrybutów Ai oraz zbioru próbek trenujących Ti
    4. Powtarzaj kroki od 1. k razy
    5. W przypadku klasyfikacji przeprowadź głosowanie większościowe
        5*. W przypadku regresji uśrednij wynik wszystkich drzew

Pseudokod Lasu Losowego

Gdzie:

$n$ - Rozmiar próby bootstrap zbioru próbek trenujących $T$, zazwyczaj
$n = \lfloor \frac{2}{3}|T| \rfloor$

$m$ - Rozmiar próby bootstrap zbioru atrybutów $A$, zazwyczaj
$m = \lfloor \sqrt{|A|} \rfloor$

$k$ - Ilość drzew w lesie, zazwyczaj od kilkuset do kilku tysięcy

Budowanie drzew w lesie losowym może również być dostosowywane pod
względem wyboru metody przycinania, i innych modyfikacji. Zazwyczaj
jednak używa się metody CART.

Plan eksperymentów
==================

W ramach badań będziemy porównywali metryki naszej implementacji
algorytmu z algorytmem ID3 z biblioteki scikit
[dokumentacja](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart).
Będziemy brać pod uwagę następujące metryki przy porównywaniu
implementacji algorytmów:

$$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$
$$Precision = \frac{TP}{TP+FP}$$ 
$$Recall = \frac{TP}{TP+FN}$$
$$F1 = \frac{2*Precision*Recall}{Precision+Recall} = \frac{2*TP}{2*TP+FP+FN}$$

Dodatkowo, ponieważ spodziewamy się, że nasza implementacja algorytmu
będzie cechowała się większą złożonością obliczeniową niż implementacja
z biblioteki scikit, piątym i szóstym parametrem służącym do porównania
wydajności algorytmów będą czas potrzebny treningu oraz czas predykcji.

W ocenie modeli będziemy także wspierać się analizą krzywej ROC.

Mierząc parametry modeli, w zależności od jakości i liczności zbiorów
danych będziemy stosować k-krotną walidację krzyżową, oraz w przypadku
lasu losowego tak zwany Out-Of-Bag Error (OOBe).

Zbiory danych do badań
======================

Do badań zastosujemy następujące zbiory danych:

1.  **Heart Attack Analysis & Prediction Dataset** (klasyfikacja
    binarna) - [Heart Attack Analysis & Prediction Dataset
    \[Kaggle\]](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/data)\
    Detekcja ataku serca na bazie atrybutów m.in. takich jak wiek, płeć,
    typ bólu, spoczynkowe ciśnienie krwi.

2.  **Parkinson Disease Detection** (klasyfikacja binarna) - [Parkinson
    Disease Detection
    \[Kaggle\]](https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection)\
    Detekcja choroby parkinsona na podstawie atrybutów dotyczących mowy,
    takich jak m.in. średnia/maksymalna/minimalna częstotliwość
    podstawowa wokalu, zmiany częstotliwości podstawowej.

3.  **Wine Quality Dataset** (klasyfikacja hierarchiczna) - [Wine
    Quality Dataset
    \[Kaggle\]](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)\
    Przypisanie klasy jakości wina na podstawie atrybutów takich jak
    ustalona kwasowość, lotna kwasowość, kwas cytrynowy, cukier
    resztkowy, chlorki, wolny dwutlenek siarki.

4.  **Date Fruit Dataset** (klasyfikacja wieloklasowa) - [Date Fruit
    Dataset
    \[Kaggle\]](https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets)\
    Przypisanie gatunku owoca na podstawie parametrów owoca uzyskanych
    ze zdjęcia (Computer Vision) kolor, długość, średnica i kształt.

5.  **Rice MSC Dataset** (klasyfikacja wieloklasowa) - [Rice MSC Dataset
    \[Kaggle\]](https://www.kaggle.com/datasets/muratkokludataset/rice-msc-dataset)
    Klasyfikacja gatunku ziarnka ryżu na podstawie atrybutów uzyskanych
    za pomocą techniki przetwarzania obrazu. Atrybuty brane pod uwagę to
    m.in. okrągłość, zwartość, współczynnik kształtu i ekscentryczność.
