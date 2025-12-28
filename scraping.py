from icrawler.builtin import BingImageCrawler
import os

def scrape_bing(query, folder, max_num=50):
    os.makedirs(folder, exist_ok=True)

    crawler = BingImageCrawler(
        storage={'root_dir': folder}
    )

    crawler.crawl(
        keyword=query,
        max_num=max_num,
        filters={
            'type': 'photo',
            'size': 'medium'
        }
    )

    print(f"[DONE] Selesai download: {query}")


# ---------------- RUN ---------------- #

scrape_bing("normal skin face woman", "dataset/normal", 100)
scrape_bing("oily skin face woman", "dataset/oily", 100)
scrape_bing("dry skin face woman", "dataset/dry", 100)
scrape_bing("combination skin face woman", "dataset/combination", 100)

print("SEMUA SELESAI!")
