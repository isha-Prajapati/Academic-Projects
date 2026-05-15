# Ukulele Data Analysis System

A Python-based data analysis and visualization system developed for analyzing song, play history, and request datasets for the Ukulele Tuesday music group.

---

## Project Overview

This project was developed as part of the UCD MSc Business Analytics MIS41110 Project. The application allows users to load, filter, analyze, and visualize music-related datasets stored in CSV files.

The system provides an interactive way to:
- Query song-related data
- Filter records based on multiple criteria
- Analyze song play history
- Generate visual insights using charts and graphs

The program focuses on clean data handling, user-friendly interaction, and statistical visualization.

---

## Features

### Data Loading & Validation
- Reads multiple CSV datasets:
  - `tabdb.csv`
  - `playdb.csv`
  - `requestdb.csv`
- Validates file structure and handles invalid or missing data gracefully

### Data Querying
Users can filter songs based on:
- Song title
- Artist
- Year
- Language
- Difficulty level
- Source
- Gender
- Date ranges
- Play frequency

### Data Visualization
The program generates several visualizations including:

- Histogram of songs by difficulty level
- Histogram of song duration
- Bar chart of songs by language
- Bar chart of songs by source
- Bar chart of songs by decade
- Cumulative line chart of songs played over time
- Pie chart of songs by gender

### Data Privacy
- Personal information such as tabber names is not displayed or processed

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- CSV File Handling

---

## Project Structure

```text
Ukulele-Tuesday-Analysis/
│
├── Ukulele_Interface.py
└── README.md
