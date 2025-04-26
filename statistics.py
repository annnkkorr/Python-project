# Статистика
import pandas as pd
import numpy as np

def calculate_summary_statistics(individual_measurements):
    if individual_measurements.empty:
        return pd.DataFrame()

    summary = pd.DataFrame({
        'max_size_mean': [np.mean(individual_measurements['Max_Size'])],
        'max_size_std': [np.std(individual_measurements['Max_Size'])],
        'min_size_mean': [np.mean(individual_measurements['Min_Size'])],
        'min_size_std': [np.std(individual_measurements['Min_Size'])]
    })

    return summary

def create_summary_table(individual_data):
    if individual_data.empty:
        return pd.DataFrame()

    summary_data = individual_data.groupby('Image').apply(calculate_summary_statistics).reset_index()
    summary_data = summary_data.droplevel(level=1)

    summary_data['max_size'] = summary_data['max_size_mean'].astype(str) + ' ± ' + summary_data['max_size_std'].astype(str)
    summary_data['min_size'] = summary_data['min_size_mean'].astype(str) + ' ± ' + summary_data['min_size_std'].astype(str)

    summary_table = summary_data[['Image', 'max_size', 'min_size']].rename(columns={'Image': 'Название фотографии', 'max_size': 'Максимальный размер споры', 'min_size': 'Минимальный размер споры'})

    return summary_table
def save_to_excel(df, filename):
    df.to_excel(filename, index=False)

def process_data(input_excel="individual_measurements.xlsx", output_excel="summary_statistics.xlsx"):
    try:
        individual_data = pd.read_excel(input_excel)
    except FileNotFoundError:
        print(f"Ошибка: файл {input_excel} не найден.")
        return

    # Создаем итоговую таблицу
    summary_table = create_summary_table(individual_data)

    # Сохраняем таблицу в Excel
    if not summary_table.empty:
        save_to_excel(summary_table, output_excel)
        print(f"Итоговая таблица {output_excel} успешно создана.")
    else:
        print("Итоговая таблица пуста. Проверьте входные данные.")

if __name__ == '__main__':
    process_data()