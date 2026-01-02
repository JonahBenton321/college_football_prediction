from playwright.sync_api import sync_playwright
import numpy as np

# The purpose this script is to go to the schedule page for naiastats and scrape every link to every box score for a given year

# years the scraper will cover
years = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

# Variable which will hold the scraped links
data = []

with sync_playwright() as p:
    for year in years:
        # url which host boxscore links
        url = f"https://naiastats.prestosports.com/sports/fball/{year}/schedule"
        try:
            # Creation of browser
            browser = p.firefox.launch()
            page = browser.new_page()

            # Timeout is set to 2 minutes as the schedule page often takes over a minute to load
            page.set_default_navigation_timeout(1200000)

            # Connect to the url and waits for the network to be idle
            page.goto(url)
            page.wait_for_load_state("networkidle")

            # Collect all links on the page
            all_links = page.query_selector_all("a")

            # Finds only 'boxscore' links and adds them to the data array
            for a in all_links:
                link = a.get_attribute('href')
                if link is not None:
                    if 'boxscores' in link.strip('/'):
                        data.append(link)
        except Exception as e:
            print(e)
        finally:

            # Closes browsers
            browser.close()

# Converts data to numpy and saves to a file
links_numpy = np.array(data)
np.save('../../Data/links/links_to_games.npy', links_numpy)