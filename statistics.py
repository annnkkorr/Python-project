# Статистика
           max_sizes.append(max_size)
                min_sizes.append(min_size)
        if max_sizes and min_sizes:
            self.results.append({
                'Image': filename,
                'Mean_Max_Size': f"{np.mean(max_sizes):.2f} ± {np.std(max_sizes):.2f}",
                'Median_Max_Size': np.median(max_sizes),
                'Mean_Min_Size': f"{np.mean(min_sizes):.2f} ± {np.std(min_sizes):.2f}",
                'Median_Min_Size': np.median(min_sizes),
                'Spore_Count': len(max_sizes)
            })

    def run(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        if not image_files:
            print("❌ В папке нет изображений для обработки.")
            return
        
        print(f"🔍 Найдено {len(image_files)} файлов. Начинаю обработку...")

        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda f: self.process_image(os.path.join(self.input_dir, f), f), image_files),
                      total=len(image_files), desc="Processing"))

        if not self.individual_measurements:
            print("❌ Не удалось обработать ни одной споры. Проверьте качество изображений.")
            return

        individual_df = pd.DataFrame(self.individual_measurements)
        summary_df = pd.DataFrame(self.results)

        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            individual_df.to_excel(writer, sheet_name='Individual_measurements', index=False)
            summary_df.to_excel(writer, sheet_name='Summary_statistics', index=False)

        print(f"✅ Готово. Результаты сохранены в файл: {self.output_file}")
