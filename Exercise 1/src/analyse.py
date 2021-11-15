class ERROR(Enum):
	TP=auto()
	FP=auto()
	FN=auto()
	TN=auto()
	
def getError(prediction, truth):
	if prediction and truth:
		return ERROR.TP
	if prediction and not truth:
		return ERROR.FP
	if not prediction and truth:
		return ERROR.FN
	if not prediction and not truth:
		return ERROR.TN

def descriptive_plot(data):
    std_scaler = StandardScaler()
    # plot box plot of every variable describing mean
    plt.rcParams['figure.figsize'] = [20, 3]
    data.DataFrame(std_scaler.fit_transform(data.iloc[:, 2:12]), columns=data.iloc[:, 2:12].columns).boxplot()