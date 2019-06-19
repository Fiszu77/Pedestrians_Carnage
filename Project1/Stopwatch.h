class Stopwatch {
private:
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
public:
	Stopwatch();
	void startTimer();
	void getTimeWithMessage();
	double getTime();
};