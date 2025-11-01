import pandas as pd
import numpy as np
from typing import Union, Optional


class IWP:

    def __init__(self, max_iterations: int = 10, random_state: Optional[int] = None):
        self.max_iterations = max_iterations
        self.weights = None  # Bude uchovávať najlepšie nájdené váhy
        self.random_state = random_state
        self.feature_names_ = None
        self.iteration_scores = []  # Na ukladanie skóre pre každú iteráciu

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        # Konvertovať vstupy na numpy polia, ak sú to pandas objekty
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist() # Uloženie názvov atribútov
            X = X.values
        else:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])] # Namapovanie názvov atribútov na x1, x2...

        if isinstance(y, pd.Series):
            y = y.values # Uloženie názvov atribútov

        # Uistíme sa, že y je binárne
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError(f"Cieľ y musí byť binárny, dostali sme {len(unique_y)} unikátnych hodnôt")

        # Mapovanie y na 0, 1 ak je to potrebné
        if not np.all(np.isin(unique_y, [0, 1])):
            y_mapping = {val: idx for idx, val in enumerate(unique_y)}
            y = np.array([y_mapping[val] for val in y])

        n_samples, n_features = X.shape

        # Pridať stĺpec -1 do X pre prahový člen (w0)
        X_aug = np.hstack([-np.ones((n_samples, 1)), X])
        n_features_aug = n_features + 1

        # Nastavenie random seed, ak je špecifikované
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Inicializovať váhy v rozsahu (-1, 1) na 2 destatinné miesta
        H = np.round(np.random.uniform(-1, 1, n_features_aug), 2)
        BEST = H.copy()

        best_score = self._calculate_ratio_score(X_aug, y, H)

        # Uložiť počiatočné váhy a skóre
        self.iteration_scores.append({
            'iteration': 0,
            'score': best_score,
            'weights': H.copy()
        })

        print(f"Počiatočné váhy: {H}")
        print(f"Počiatočné skóre: {best_score:.4f}")

        count = self.max_iterations
        perfect_solution_found = False

        print("\nSpúšťam iterácie IWP:")
        print("-" * 50)

        iteration = 1
        while count > 0 and not perfect_solution_found:
            print(f"\nIterácia {iteration}/{self.max_iterations}")
            print("-" * 30)

            weight_changes = []
            feature_improvements = []

            for k in range(n_features_aug):  # Pre každý atribút (vrátane prahu)
                feature_name = "prah" if k == 0 else self.feature_names_[k-1]
                print(f"  Spracovávam atribút: {feature_name} (index {k})")

                old_weight = H[k]

                # Vypočítať hodnoty Ukj pre každý príklad
                ukj_values = self._calculate_ukj_values(X_aug, y, H, k)

                # Preskočiť, ak sú všetky hodnoty Ukj nulové (stáva sa, keď má atribút samé nuly)
                if np.all(ukj_values == 0):
                    print(f"    Preskakujem atribút {feature_name} - všetky hodnoty Ukj sú nulové")
                    continue

                # Zoradiť hodnoty Ukj podľa absolútnej hodnoty (zostupne)
                sorted_indices = np.argsort(np.abs(ukj_values))[::-1]
                sorted_ukjs = ukj_values[sorted_indices]

                # Vypočítať negatívne priemery párov
                w_candidates = self._calculate_negative_average_pairs(sorted_ukjs)

                if len(w_candidates) == 0:
                    print(f"    Nenašli sa žiadni kandidáti na váhy pre atribút {feature_name}")
                    continue

                # Vyhodnotiť každého kandidáta na váhu - použitím hodnotenia ratio
                best_k_score = 0
                best_k_weight = H[k]

                for w_candidate in w_candidates:
                    H_candidate = H.copy()
                    H_candidate[k] = w_candidate

                    score = self._calculate_ratio_score(X_aug, y, H_candidate)

                    if score > best_k_score or (score == best_k_score and abs(w_candidate) < abs(best_k_weight)):
                        best_k_score = score
                        best_k_weight = w_candidate

                # Aktualizovať váhu pre tento atribút
                old_score = self._calculate_ratio_score(X_aug, y, H)
                H[k] = best_k_weight
                new_score = self._calculate_ratio_score(X_aug, y, H)

                weight_changes.append({
                    'feature': feature_name,
                    'old_weight': old_weight,
                    'new_weight': best_k_weight,
                    'score_change': new_score - old_score
                })

                feature_improvements.append(new_score - old_score)

                print(f"    Váha aktualizovaná: {old_weight:.4f} -> {best_k_weight:.4f} (Skóre: {old_score:.4f} -> {new_score:.4f})")

                # Skontrolovať, či sme našli dokonalé riešenie
                current_score = new_score
                if current_score == 1.0:
                    print(f"    Dokonalé riešenie nájdené! Skóre: {current_score:.4f}")
                    self.weights = H
                    perfect_solution_found = True
                    break

                # Aktualizovať BEST, ak je aktuálne riešenie lepšie
                if current_score > best_score:
                    BEST = H.copy()
                    best_score = current_score
                    print(f"    Nové najlepšie skóre: {best_score:.4f}")

            # Uložiť výsledky tejto iterácie
            current_score = self._calculate_ratio_score(X_aug, y, H)
            self.iteration_scores.append({
                'iteration': iteration,
                'score': current_score,
                'weights': H.copy()
            })

            # Vypísať súhrn zmien váh
            print("\n  Súhrn zmien váh:")
            for change in weight_changes:
                print(f"    {change['feature']}: {change['old_weight']:.4f} -> {change['new_weight']:.4f} (Zmena skóre: {change['score_change']:.4f})")

            # Vypísať najviac vylepšené atribúty
            if feature_improvements:
                most_improved_idx = np.argmax(feature_improvements)
                most_improved = weight_changes[most_improved_idx]['feature']
                print(f"\n  Najviac vylepšený atribút: {most_improved} (Zvýšenie skóre: {max(feature_improvements):.4f})")

            print(f"\n  Koniec iterácie {iteration}")
            print(f"  Aktuálne skóre: {current_score:.4f}")

            count -= 1
            iteration += 1

        self.weights = BEST

        # Vypísať finálny súhrn
        print("\n" + "=" * 50)
        print("Trénovanie IWP dokončené")
        print("=" * 50)
        final_score = self._calculate_ratio_score(X_aug, y, BEST)
        print(f"Konečné skóre: {final_score:.4f}")
        print(f"Dokončené iterácie: {iteration - 1}")
        print(f"Dokonalé riešenie nájdené: {perfect_solution_found}")

        # Vypísať tabuľku skóre iterácií
        self._print_score_table()

        return self

    def _calculate_ratio_score(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        n_samples = X.shape[0]

        # Vytvoriť rozhodovaciu funkciu
        decision_values = X.dot(weights)
        classifications = (decision_values >= 0).astype(int)

        # Spočítať správne predikcie v každej skupine
        correct_classifications = np.sum(classifications == y)
        ratio_score = correct_classifications / n_samples

        return ratio_score

    def _print_score_table(self):
        print("\nTabuľka skóre iterácií:")
        print("-" * 50)
        print(f"{'Iterácia':<10} {'Skóre':<10} {'Zlepšenie':<15}")
        print("-" * 50)

        prev_score = None
        for entry in self.iteration_scores:
            iteration = entry['iteration']
            score = entry['score']

            if prev_score is not None:
                improvement = score - prev_score
                print(f"{iteration:<10} {score:.4f}     {improvement:+.4f}")
            else:
                print(f"{iteration:<10} {score:.4f}     {'--':<15}")

            prev_score = score

        print("-" * 50)
        total_improvement = self.iteration_scores[-1]['score'] - self.iteration_scores[0]['score']
        print(f"Celkové zlepšenie: {total_improvement:.4f}")
        print("-" * 50)

    def _calculate_ukj_values(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
        n_samples = X.shape[0]
        ukj_values = np.zeros(n_samples)

        for j in range(n_samples):
            # Preskočiť, ak je xkj nula, aby sa zabránilo deleniu nulou
            if X[j, k] == 0:
                continue

            # Vypočítať Vj (rozhodovacia hodnota pre príklad j)
            vj = np.dot(weights, X[j])

            # Vypočítať Ukj podľa vzorca
            ukj_values[j] = vj / X[j, k] - weights[k]

        return ukj_values

    def _calculate_negative_average_pairs(self, sorted_ukjs: np.ndarray) -> np.ndarray:
        # Odfiltrovať nuly
        sorted_ukjs = sorted_ukjs[sorted_ukjs != 0]

        if len(sorted_ukjs) <= 1:
            return np.array([])

        n = len(sorted_ukjs) - 1
        negative_avgs = np.zeros(n)

        for i in range(n):
            negative_avgs[i] = -((sorted_ukjs[i] + sorted_ukjs[i+1]) / 2)

        return negative_avgs

    def get_iteration_history(self):
        if not self.iteration_scores:
            return None

        data = []
        for entry in self.iteration_scores:
            row = {
                'iteration': entry['iteration'],
                'score': entry['score'],
                'threshold': -entry['weights'][0]
            }

            # Pridať váhy atribútov
            for i, name in enumerate(self.feature_names_):
                row[name] = entry['weights'][i+1]

            data.append(row)

        return pd.DataFrame(data)

    def __str__(self) -> str:
        if self.weights is None:
            return "IWP(nenatrénovaný)"

        threshold = -self.weights[0]  # w0 je záporná hodnota prahu
        feature_weights = self.weights[1:]

        formula_parts = []
        for i, (weight, name) in enumerate(zip(feature_weights, self.feature_names_)):
            if abs(weight) > 1e-6:  # Zahŕňa len nenulové váhy
                sign = "+" if weight > 0 else ""
                formula_parts.append(f"{sign}{weight:.3f}*{name}")

        formula = " ".join(formula_parts)
        return f"IWP: {formula} >= {threshold:.3f}"


def run_iwp_on_csv(csv_file, target_column, max_iterations=3, random_state=42):
    data = pd.read_csv(csv_file)

    # Kontrola, či existuje cieľový stĺpec
    if target_column not in data.columns:
        raise ValueError(f"Cieľový stĺpec '{target_column}' nebol nájdený v CSV súbore")

    # Ak cieľ nie je binárny, konvertovať vysoké hodnoty na 1 a nízke hodnoty na 0
    if len(data[target_column].unique()) != 2:
        print(f"Cieľový stĺpec '{target_column}' nie je binárny. Konvertujem na binárny pomocou mediánu.")
        threshold = data[target_column].median()
        data[target_column] = (data[target_column] >= threshold).astype(int)

    # Rozdeliť atribúty a cieľ
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Vytvoriť a natrénovať model IWP
    iwp = IWP(max_iterations=max_iterations, random_state=random_state)
    iwp.fit(X, y)

    # Výpočet korektnosti
    X_aug = np.hstack([-np.ones((X.shape[0], 1)), X.values if isinstance(X, pd.DataFrame) else X])
    decision_values = X_aug.dot(iwp.weights)
    classifications = (decision_values >= 0).astype(int)
    ratio = np.sum(classifications == y) / len(y)

    print(f"\nPresnosť: {ratio:.4f}")
    print(f"IWP model: {iwp}")

    # Získanie iteračnej histórie
    iteration_history = iwp.get_iteration_history()

    return iwp, ratio, iteration_history


# Použitie na datasete heart_disease.csv
if __name__ == "__main__":
    csv_file = "stroj/heart_disease.csv" # Výber dátovej množiny
    target_column = "target" # Nastavenie cieľového atribútu

    try:
        iwp_model, ratio, iteration_history = run_iwp_on_csv(
            csv_file, target_column, max_iterations=3,
            random_state=32
        )

        print("\nHistória iterácií:")
        print(iteration_history)

    # Ošetrenie chybných vstupov
    except FileNotFoundError:
        print(f"CSV súbor '{csv_file}' nebol nájdený. Prosím, zadajte správnu cestu.")
    except Exception as e:
        print(f"Chyba: {e}")