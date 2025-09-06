class DataLoader:
    def __init__(self, source):
        self.source = source

    def load(self):
        if self.source.endswith('.csv'):
            return self.load_csv()
        elif self.source.endswith('.xlsx'):
            return self.load_excel()
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    def load_csv(self):
        import pandas as pd
        return pd.read_csv(self.source)

    def load_excel(self):
        import pandas as pd
        return pd.read_excel(self.source)