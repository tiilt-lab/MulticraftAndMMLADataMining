set /p val="0 for quiz processing, 1 for build processing: "
cd src 
if %val%==0 (
	python quadrant_gaze_detector.py
) else (
	python portfolio_identifier.py
)
cd ..