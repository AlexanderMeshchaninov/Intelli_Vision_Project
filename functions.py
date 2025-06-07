import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from inspect import signature

def load_descriptors(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_descriptors(df, desc_name, save_folder='Data'):
    """Сохраняет DataFrame df в формате pkl по заданному имени дескриптора.
    
    Args:
        df (pd.DataFrame): DataFrame для сохранения.
        desc_name (str): Название дескриптора.
        save_folder (str): Папка для сохранения файла.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    save_path = os.path.join(save_folder, f'{desc_name.replace(" ", "_").lower()}.pkl')
    df.to_pickle(save_path)
    print(f"===> Сохранено: {save_path}")
    
def dataset_overview(df, name="DataFrame"):
    print(f"\n--- Обзор: {name} ---")
    print(f"Размер (строк, столбцов): {df.shape}")
    
    print("\nИнформация о типах данных:")
    print(df.info())
    
    print("\nСтатистика по числовым столбцам:")
    print(df.describe())
    
    print("\nКоличество пропущенных значений по столбцам:")
    print(df.isnull().sum())
    
    print(f"\nКоличество дублирующихся строк: {df.duplicated().sum()}")
    print('--' * 20)

def get_descriptor_length(df, name):
    sample_vector = df['features'].iloc[0]
    print(f"{name}: длина дескриптора = {len(sample_vector)}")
    print('--' * 20)

def split_and_save_filtered_columns(df, 
                                    desc_name, 
                                    columns_to_remove=None, 
                                    save_folder='Data'
                                ):
    """
    Удаляет указанные столбцы из df и сохраняет результат.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        desc_name (str): Название дескриптора (для имени файла).
        columns_to_remove (list): Список названий столбцов для удаления.
        save_folder (str): Папка для сохранения.
    """
    columns_to_remove = columns_to_remove or []

    # Лог до удаления
    print(f"\n=== Обработка дескриптора: {desc_name} ===")
    print(f"Изначально столбцов: {len(df.columns)}")
    print(f"Будет удалено столбцов: {columns_to_remove}")

    # Проверка что все столбцы есть в df
    missing_cols = [col for col in columns_to_remove if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Ошибка: указанные столбцы не найдены в df: {missing_cols}")

    # Удаление столбцов
    df_filtered = df.drop(columns=columns_to_remove)

    # Лог после удаления
    print(f"Осталось столбцов после удаления: {len(df_filtered.columns)}")

    # Сохраняем
    save_descriptors(df_filtered, desc_name, save_folder)

def clustering_evaluation(df,
                          desc_name='Descriptor',
                          cluster_model_cls=None,
                          cluster_model_params=None,
                          k_range=range(2, 10),
                          metrics=None,
                          fixed_k=None,
                          selected_pca_cols=None
                        ):
    """
    Оценивает кластеризацию для выбранных pca_* колонок DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame с колонками pca_*
        desc_name (str): Имя дескриптора
        cluster_model_cls (type): Класс модели кластеризации (например, KMeans)
        cluster_model_params (dict): Параметры модели
        k_range (range): Диапазон k для подбора (если fixed_k не задан)
        metrics (list of tuples): [(метрика, имя_метрики), ...]
        fixed_k (int): Фиксированное k (если задано)
        selected_pca_cols (list): Список колонок pca_* для кластеризации
    """
    cluster_model_params = cluster_model_params or {}
    metrics = metrics or [(silhouette_score, 'silhouette_score')]
    
    # По умолчанию ищем все pca_* колонки
    if selected_pca_cols is None:
        selected_pca_cols = [col for col in df.columns if col.startswith('pca_')]
    
    uses_k = 'n_clusters' in signature(cluster_model_cls).parameters
    
    print(f"\n=== Оценка кластеризации: {desc_name} ===")
    
    for col in selected_pca_cols:
        print(f"\n  Колонка: {col}")
        X = np.vstack(df[col].values)
        
        for metric_func, metric_name in metrics:
            best_k, best_score = None, None
            k_values = [fixed_k] if fixed_k else k_range
            
            if uses_k:
                for k in k_values:
                    params = cluster_model_params.copy()
                    params['n_clusters'] = k
                    model = cluster_model_cls(**params)
                    labels = model.fit_predict(X)
                    
                    if len(set(labels)) <= 1:
                        continue
                    
                    score = metric_func(X, labels)
                    if (best_score is None or
                        (metric_name == 'davies_bouldin_score' and score < best_score) or
                        (metric_name != 'davies_bouldin_score' and score > best_score)):
                        best_k, best_score = k, score
                
                if best_score is not None:
                    print(f"{metric_name}: Лучшее k = {best_k}, score = {best_score:.4f}")
                else:
                    print(f"{metric_name}: Недостаточно кластеров")
            
            else:
                model = cluster_model_cls(**cluster_model_params)
                labels = model.fit_predict(X)
                mask = labels != -1
                
                print(f"Кластеры: {Counter(labels)}")
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    score = metric_func(X[mask], labels[mask])
                    print(f"{metric_name}: score = {score:.4f}")
                else:
                    print(f"{metric_name}: Недостаточно данных для оценки")

def perform_pca_and_save_df(df,
                            desc_name='Descriptor',
                            selected_scalers=None,  # список скейлеров для обработки (['robust_scaled', ...])
                            explained_variance_threshold=0.95,
                            max_components=None,   # новое: максимальное число компонент (None — без ограничения)
                            save_folder='Data',
                            draw_plots=True
                        ):
    """
    Выполняет PCA для выбранных скейлеров в df и сохраняет обратно df с новыми колонками pca_*.

    Args:
        df (pd.DataFrame): загруженный DataFrame (scaled_* колонки уже есть)
        desc_name (str): имя дескриптора для сохранения
        selected_scalers (list): список колонок, по которым делать PCA. Если None — все.
        explained_variance_threshold (float): порог накопленной дисперсии
        max_components (int or None): максимальное число компонент PCA
        save_folder (str): куда сохранять результат
        draw_plots (bool): рисовать ли графики explained variance
    """
    print(f"\n=== Обработка дескриптора: {desc_name} ===")
    
    all_scaler_cols = ['standard_scaled', 'minmax_scaled', 'robust_scaled']
    scaler_cols = all_scaler_cols if selected_scalers is None else selected_scalers
    
    for col in scaler_cols:
        if col not in df.columns:
            print(f"Пропущено {col} — колонки нет в df")
            continue
        
        print(f"  PCA для {col}")
        X = np.vstack(df[col].values)
        
        pca_full = PCA().fit(X)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cum_var >= explained_variance_threshold) + 1
        
        if max_components is not None:
            n_components = min(n_components, max_components)
        
        print(f"-> Оптимальное число компонент: {n_components}")
        
        X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
        df[f'pca_{col}'] = list(X_pca)
        
        if draw_plots:
            plt.figure(figsize=(8, 5))
            plt.plot(cum_var, marker='.')
            plt.axhline(y=explained_variance_threshold, color='r', linestyle='--')
            plt.axvline(x=n_components - 1, color='g', linestyle='--')
            plt.title(f"{desc_name} — {col} — Explained Variance")
            plt.xlabel('Число компонент')
            plt.ylabel('Накопленная дисперсия')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    save_descriptors(df, f'pca_scaled_{desc_name}', save_folder=save_folder)

def plot_samples_images(data, 
                        cluster_col, 
                        cluster_label, 
                        descriptor_name="Unknown Descriptor",  # добавил параметр
                        nrows=3,
                        ncols=3, 
                        figsize=(12, 5), 
                        base_dir="Data/raw_data"
                    ):
    """Функция для визуализации изображений из указанного кластера."""
    
    # Фильтруем по кластеру
    samples_indexes = np.array(data[data[cluster_col] == cluster_label].index)
    np.random.shuffle(samples_indexes)
    
    paths = data.loc[samples_indexes, 'image_path']
    
    # График
    fig, axes = plt.subplots(nrows, ncols)
    fig.set_size_inches(*figsize)
    fig.suptitle(
        f"Descriptor: {descriptor_name} | Cluster column: {cluster_col} | Cluster #{cluster_label}",
        fontsize=16
    )
    
    for i in range(nrows):
        for j in range(ncols):
            path_idx = i * ncols + j
            # Безопасно выбираем ось
            ax = axes[i, j] if nrows > 1 and ncols > 1 else axes[max(i, j)]
            
            if path_idx >= len(paths):
                ax.axis('off')
                continue
            
            # Приводим путь к полному (с учётом base_dir)
            path = paths.iloc[path_idx]
            full_path = os.path.join(base_dir, path.replace("\\", "/"))
            
            try:
                img = plt.imread(full_path)
                ax.imshow(img)
            except Exception as e:
                print(f"Ошибка при чтении изображения: {full_path} ({e})")
                ax.imshow(np.zeros((10, 10, 3)))  # Пустая картинка
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_outliers(df, 
                  base_dir, 
                  descriptor, 
                  cluster_col='cluster_DBSCAN'
                  ):

    outliers = df[df[cluster_col] == -1]
    if outliers.empty:
        print(f"Выбросов не найдено для {descriptor}")
        return
    
    print(f"Выводим выбросы для {descriptor}")
    sample = outliers.sample(min(9, len(outliers)))
    plt.figure(figsize=(12, 6))
    
    for i, (_, row) in enumerate(sample.iterrows()):
        img_path = f"{base_dir}/{row['image_path'].replace('\\', '/')}"
        try:
            img = plt.imread(img_path)
            plt.subplot(3, 3, i+1)
            plt.imshow(img)
            plt.axis('off')
        except:
            continue
    plt.suptitle(f"Выбросы: {descriptor}")
    plt.show()