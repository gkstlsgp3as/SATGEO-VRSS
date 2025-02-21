# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:39:52 2024

@author: Soyeon Park

Description: Download ECMWF forecast results using API

How to use: python ECMWF_API.py --save <savepath> --date <forecast date to download>
Example usage: python ECMWF_API.py -s D:\Extract_spectra\ECMWF_forecast -d 20240311 20240312

"""

import asyncio
import datetime
import os
from typing import List

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from uvicorn.config import logger

from app.config.settings import settings
from app.models.ecmwf_collect_hist import EcmwfCollectHist
from app.service import ecmwf_collect_hist_service

# 다운로드 큐 설정
CONCURRENT_WORKERS = 10
TIMEOUT = 300.0
CHUNK_SIZE = 256 * 1024  # 청크 크기 8192


async def download_file(client: httpx.AsyncClient, file_url: str, file_path: str, retries: int = 2, initial_delay: int = 5):
    attempt = 0
    while attempt < retries:
        try:
            async with client.stream("GET", file_url) as response:
                response.raise_for_status()

                with open(file_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                        f.write(chunk)

                logger.info(f'{os.path.basename(file_path)} downloaded successfully.')
            return

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            attempt += 1
            logger.info(f"Failed to download {file_url} on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                await asyncio.sleep(initial_delay * attempt)

        except Exception as e:
            logger.info(f"Unexpected error while downloading {file_url}: {e}")
            break

    logger.error(f"Failed to download {file_url} after {retries} attempts.")


async def worker(queue: asyncio.Queue, client: httpx.AsyncClient):
    while True:
        file_url, file_path = await queue.get()
        try:
            await download_file(client, file_url, file_path)
        except Exception as e:
            logger.info(f"Worker failed to download {file_url}: {e}")
        finally:
            queue.task_done()


async def download_grib_files(url_template: str, date: str, output_dir: str, time: str, types: List[str]):
    date_fmt = f"{date[:4]}_{date[4:6]}_{date[6:]}"
    target_dir = os.path.join(output_dir, date_fmt, time)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    queue = asyncio.Queue()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for t in types:
            url = url_template.format(date=date, time=time, file_type=t)
            try:
                response = await client.get(url)
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.error(f"Failed to fetch URL: {url} with error: {e}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a')

            for link in links:
                href = link.get('href')
                if href and href.endswith('.grib2'):
                    file_name = os.path.basename(href)
                    file_url = url + file_name
                    file_path = os.path.join(target_dir, file_name)
                    await queue.put((file_url, file_path))

        # Start workers
        workers = [asyncio.create_task(worker(queue, client)) for _ in range(CONCURRENT_WORKERS)]
        await queue.join()  # Wait until all files are downloaded

        for w in workers:
            w.cancel()

        logger.info(f'All files have been downloaded for {date} at {time}.')


# Download GRIB files for a list of dates and predefined times.
#
#     Args:
#     - transaction_id (str): transaction_id id.
#     - dates (List[str]): List of date strings in the format YYYYMMDD.
async def lists_by_dates(db: Session, collect_history: EcmwfCollectHist, dates: List[str]):
    output_dir = settings.W01_OUTPUT_PATH

    url_template = 'https://data.ecmwf.int/forecasts/{date}/{time}/ifs/0p25/{file_type}/'
    times_types = {
        '00z': ['oper', 'wave'],
        '12z': ['oper', 'wave'],
        '06z': ['scda', 'scwv'],
        '18z': ['scda', 'scwv']
    }

    download_tasks = [download_grib_files(url_template, date, output_dir, time, types) for date in dates for time, types in times_types.items()]
    await asyncio.gather(*download_tasks)

    collect_history.status = "completed"
    collect_history.updated_at = datetime.datetime.now()
    logger.info(f"====== updated: {collect_history.updated_at} ======")
    ecmwf_collect_hist_service.update_ecmwf_collect_history(db, collect_history)


def get_task_status(db: Session, task_id: str, run_id: str, transaction_id: str):
    return ecmwf_collect_hist_service.get_ecmwf_collect_history(db, task_id, run_id, transaction_id)
