# src/crmstudio/monitoring/alerts.py

class Alert:
    def __init__(self, threshold):
        self.threshold = threshold
        self.alerts_triggered = []

    def check_metric(self, metric_value, metric_name):
        if metric_value > self.threshold:
            self.trigger_alert(metric_name, metric_value)

    def trigger_alert(self, metric_name, metric_value):
        alert_message = f"Alert: {metric_name} has exceeded the threshold! Value: {metric_value}"
        self.alerts_triggered.append(alert_message)
        print(alert_message)

    def get_alerts(self):
        return self.alerts_triggered