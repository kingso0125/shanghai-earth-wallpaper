"""Calibrated Himawari-9 Band 13, without the ambiguous browse-image palette.

HSD layout and Planck coefficients: JMA Himawari Standard Data User's Guide.
Public archive: https://registry.opendata.aws/noaa-himawari/
"""
from __future__ import annotations

import bz2
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np


def decode_segment(payload: bytes, timestamp: datetime, sequence: int):
    data = bz2.decompress(payload)
    if len(data) < 282 or data[0] != 1 or data[5] not in (0, 1):
        raise ValueError("invalid Himawari HSD header")
    endian = "<" if data[5] == 0 else ">"

    def unpack(fmt, offset):
        return struct.unpack_from(endian + fmt, data, offset)

    header_length, data_length = unpack("II", 70)
    if len(data) != header_length + data_length:
        raise ValueError("truncated Himawari segment")
    observed = datetime(1858, 11, 17, tzinfo=UTC) + timedelta(days=unpack("d", 46)[0])
    if (observed.date() != timestamp.date()
            or unpack("H", 44)[0] != timestamp.hour * 100 + timestamp.minute
            or not data[6:22].rstrip(b" \0").startswith(b"Himawari-9")):
        raise ValueError("Himawari observation does not match requested time")
    blocks, offset = {}, 0
    for number in range(1, 12):
        if offset + 5 > header_length or data[offset] != number:
            raise ValueError("invalid Himawari block order")
        length = unpack("I" if number == 10 else "H", offset + 1)[0]
        if length < 5 or offset + length > header_length:
            raise ValueError("invalid Himawari block length")
        blocks[number] = offset
        offset += length
    if offset != header_length:
        raise ValueError("invalid Himawari header size")
    bits, width, height, compression = unpack("HHHB", blocks[2] + 3)
    total, actual_sequence, first_line = unpack("BBH", blocks[7] + 3)
    if (bits, width, height, compression, total, actual_sequence, first_line) != (
            16, 5500, 550, 0, 10, sequence, (sequence - 1) * 550 + 1):
        raise ValueError("unexpected Himawari segment dimensions")
    band, wavelength, valid_bits, error, outside = unpack("HdHHH", blocks[5] + 3)
    if band != 13 or valid_bits != 12 or data_length != width * height * 2:
        raise ValueError("expected Himawari Band 13 counts")
    gain, constant, c0, c1, c2, _, _, _, light, planck, boltzmann = unpack("11d", blocks[5] + 19)
    counts = np.arange(65536, dtype=np.float64)
    radiance = counts * gain + constant
    wave = wavelength * 1e-6
    effective = (planck * light / (boltzmann * wave)
                 / np.log1p(2 * planck * light**2 / (wave**5 * np.maximum(radiance, 1e-12) * 1e6)))
    temperature = c0 + c1 * effective + c2 * effective**2
    temperature[(radiance <= 0) | (counts >= 2**valid_bits)
                | (counts == error) | (counts == outside)] = np.nan
    temperature[(temperature < 150) | (temperature > 350)] = np.nan
    projection = np.array(unpack("dIIffddd", blocks[3] + 3), dtype=np.float64)
    pixels = np.frombuffer(data, dtype=endian + "u2", count=width * height,
                           offset=header_length).reshape(height, width)
    return pixels, temperature.astype(np.float32), projection


def acquire_temperature(cache: Path, timestamp: datetime) -> Path:
    from .sources import _request

    timestamp = timestamp.astimezone(UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%MZ")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"himawari-{stamp}-band13.npz"
    if destination.exists():
        try:
            with np.load(destination, allow_pickle=False) as stored:
                if (stored["counts"].shape == (5500, 5500)
                        and stored["timestamp"].item() == timestamp.isoformat()
                        and stored["projection"].shape == (8,)
                        and stored["temperature"].shape == (65536,)):
                    return destination
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
    prefix = timestamp.strftime("AHI-L1b-FLDK/%Y/%m/%d/%H%M/HS_H09_%Y%m%d_%H%M_B13_FLDK_R20")
    deadline = time.monotonic() + 180

    def fetch(sequence):
        remaining = deadline - time.monotonic()
        if remaining < 1:
            raise OSError("Himawari temperature download budget exceeded")
        url = f"https://noaa-himawari9.s3-external-1.amazonaws.com/{prefix}_S{sequence:02d}10.DAT.bz2"
        try:
            return decode_segment(_request(url, timeout=int(remaining)), timestamp, sequence)
        except (struct.error, EOFError) as error:
            raise ValueError("invalid Himawari segment") from error

    counts = np.empty((5500, 5500), dtype=np.uint16)
    reference_projection = reference_temperature = None
    with ThreadPoolExecutor(max_workers=10) as pool:
        for index, (segment, temperature, projection) in enumerate(pool.map(fetch, range(1, 11))):
            if reference_projection is None:
                reference_projection, reference_temperature = projection, temperature
            elif (not np.array_equal(projection, reference_projection)
                  or not np.allclose(temperature, reference_temperature, equal_nan=True)):
                raise ValueError("inconsistent Himawari segment calibration")
            counts[index * 550:(index + 1) * 550] = segment
    if np.isfinite(reference_temperature[counts]).mean() < .60:
        raise ValueError("incomplete Himawari full disk")
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, counts=counts, temperature=reference_temperature,
                            projection=reference_projection, timestamp=timestamp.isoformat())
    temporary.replace(destination)
    for old in sorted(cache.glob("himawari-*-band13.npz"), reverse=True)[6:]:
        old.unlink(missing_ok=True)
    return destination


def scan_coordinates(lat: np.ndarray, lon: np.ndarray, projection: np.ndarray):
    sub_lon, cfac, lfac, coff, loff, distance, equator, polar = projection
    geocentric = np.arctan((polar / equator)**2 * np.tan(lat))
    radius = polar / np.sqrt(1 - (1 - (polar / equator)**2) * np.cos(geocentric)**2)
    delta = lon - np.deg2rad(sub_lon)
    rx = radius * np.cos(geocentric) * np.cos(delta)
    ry = radius * np.cos(geocentric) * np.sin(delta)
    rz = radius * np.sin(geocentric)
    toward = distance - rx
    x = np.arctan2(ry, toward)
    y = np.arctan2(-rz, np.sqrt(toward**2 + ry**2))
    column = coff - 1 + np.rad2deg(x) * cfac / 65536
    row = loff - 1 + np.rad2deg(y) * lfac / 65536
    visible = distance * rx >= equator**2
    return column.astype(np.float32), row.astype(np.float32), visible


def sample_temperature(path: Path, lat: np.ndarray, lon: np.ndarray):
    with np.load(path, allow_pickle=False) as stored:
        counts = stored["counts"]
        lookup = stored["temperature"]
        xf, yf, visible = scan_coordinates(lat, lon, stored["projection"])
    height, width = counts.shape
    visible &= (xf >= 0) & (xf <= width - 1) & (yf >= 0) & (yf <= height - 1)
    xf, yf = np.clip(xf, 0, width - 1), np.clip(yf, 0, height - 1)
    x0, y0 = np.floor(xf).astype(np.int32), np.floor(yf).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = xf - x0, yf - y0
    result, weight = np.zeros_like(lat), np.zeros_like(lat)
    for x, y, w in ((x0, y0, (1-wx)*(1-wy)), (x1, y0, wx*(1-wy)),
                    (x0, y1, (1-wx)*wy), (x1, y1, wx*wy)):
        kelvin = lookup[counts[y, x]]
        valid = np.isfinite(kelvin)
        result += np.where(valid, kelvin, 0) * w
        weight += valid * w
    coverage = visible & (weight > .99)
    celsius = result / np.maximum(weight, 1e-6) - 273.15
    coldness = np.clip((40 - celsius) / 140, 0, 1) * coverage
    return coldness.astype(np.float32), coverage.astype(np.float32)
