import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager


def preprocess_mct_nm(mct_nm):
    if mct_nm.startswith('('):
        mct_nm = re.sub(r'^\(.*?\)', '', mct_nm).strip()  # 괄호 묶음 제거 후 공백 제거

    if ' ' not in mct_nm:
        mct_nm += " 제주"

    return mct_nm


def get_review_data(driver):
    total_review_num = int(driver.find_element(
        By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[8]/strong[1]/span').text)
    star_mean = float(driver.find_element(
        By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[1]/div[1]/div[2]/div/div[2]/a[1]/span[1]').text)

    elements = driver.find_elements(
        By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[8]/div[2]/span[1]/span')

    good_taste = good_price = good_parking = good_facilities = good_kindness = good_mood = 0

    for n in range(len(elements)):
        review_type = driver.find_element(
            By.XPATH, f'/html/body/div[2]/div[2]/div[2]/div[7]/div[2]/span[{n}]/span[2]').text
        score = int(driver.find_element(
            By.XPATH, f'/html/body/div[2]/div[2]/div[2]/div[7]/div[2]/span[{n}]/span[3]').text)

        if review_type == "맛":
            good_taste += score
        elif review_type == "주차":
            good_parking += score
        elif review_type == "분위기":
            good_mood += score
        elif review_type == "친절":
            good_kindness += score
        elif review_type == "가성비":
            good_price += score
        elif review_type == "시설":
            good_facilities += score

    return total_review_num, star_mean, good_taste, good_price, good_parking, good_facilities, good_kindness, good_mood


def insert_data_to_db(data):
    conn = sqlite3.connect('./kakaomap.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO kakao_map (MCT_NM, ADDR, total_review_num, star_mean, good_taste, good_price, good_parking, good_facilities, good_kindness, good_mood, error_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)

    conn.commit()
    conn.close()


def fetch_mct_nm_from_db():
    conn = sqlite3.connect('./kakaomap.db')
    cursor = conn.cursor()
    cursor.execute("SELECT MCT_NM FROM kakao_map")
    mct_nms = cursor.fetchall()
    conn.close()
    return [mct_nm[0] for mct_nm in mct_nms]


driver = webdriver.Chrome()

try:
    driver.get('https://map.kakao.com/')

    mct_nms = fetch_mct_nm_from_db()

    for mct_nm in mct_nms:
        mct_nm = preprocess_mct_nm(mct_nm)
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="search.keyword.query"]')))
        search_box.clear()
        search_box.send_keys(mct_nm)

        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="search.keyword.submit"]')))
        driver.execute_script("arguments[0].scrollIntoView();", search_button)
        driver.execute_script("arguments[0].click();", search_button)

        time.sleep(2)

        try:
            print("in try")
            result_count = driver.find_element(
                By.XPATH, '//*[@id="info.search.place.cnt"]').text
            num_results = int(re.search(r'\d+', result_count).group())

            print("num_results", num_results)

            if num_results > 0:
                for n in range(num_results):
                    if num_results == 1:
                        print("num_results is 1")
                        addr_xpath = "/html/body/div[5]/div[2]/div[1]/div[6]/div[5]/ul/li/div[5]/div[2]/p[2]"
                    else:
                        addr_xpath = f"/html/body/div[5]/div[2]/div[1]/div[6]/div[5]/ul/li[{n + 1}]/div[5]/div[2]/p[2]"
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, addr_xpath))
                    )
                    addr = driver.find_element(By.XPATH, addr_xpath).text
                    print(addr)
                    print(addr)
                    print(addr)
                    print(addr)
                    print(addr)
                    print(addr)
                    print(addr)
                    addr_bunji = re.sub(r'\s*\d+-\d+', '', addr)

                    if addr_bunji in addr:
                        driver.find_element(
                            By.XPATH, f"/html/body/div[5]/div[2]/div[1]/div[6]/div[5]/ul/li[2]/div[5]/div[2]/p[2]").click()
                        time.sleep(3)

                        print("clicked!")

                        total_review_num, star_mean, good_taste, good_price, good_parking, good_facilities, good_kindness, good_mood = get_review_data(
                            driver)

                        insert_data_to_db((mct_nm, addr, total_review_num, star_mean, good_taste,
                                           good_price, good_parking, good_facilities, good_kindness, good_mood, None))

                        print(
                            f"총 리뷰 수: {total_review_num}, 평균 별점: {star_mean}, 맛: {good_taste}, 주차: {good_parking}, 친절: {good_kindness}, 가성비: {good_price}, 시설: {good_facilities}, 분위기: {good_mood}")

                        driver.back()
                        time.sleep(3)
            else:
                # 검색 결과가 없는 경우
                print("error_code 101")
                insert_data_to_db((mct_nm, None, None, None, None,
                                   None, None, None, None, None, 101))

        except Exception as e:
            # 다른 예외 처리
            print(f"예외 발생: {e}")
            insert_data_to_db((mct_nm, None, None, None, None,
                               None, None, None, None, None, 102))

except Exception as e:
    print(f"================{e}====================")
finally:
    driver.quit()
