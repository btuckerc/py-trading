"""Alerting module for trading notifications.

Supports email and Slack notifications for:
- Regime changes (especially into bear_high_vol)
- Drawdown thresholds being crossed
- Data or engine failures
- Daily trading summaries

Setup:
    Email (SMTP):
        Set environment variables:
        - ALERT_EMAIL_SMTP_HOST
        - ALERT_EMAIL_SMTP_PORT (default: 587)
        - ALERT_EMAIL_USERNAME
        - ALERT_EMAIL_PASSWORD
        - ALERT_EMAIL_FROM
        - ALERT_EMAIL_TO (comma-separated for multiple recipients)
    
    Slack:
        Set environment variable:
        - ALERT_SLACK_WEBHOOK_URL

Usage:
    from live.alerting import AlertManager, AlertLevel
    
    alerter = AlertManager()
    
    # Send regime change alert
    alerter.send_regime_change_alert(
        old_regime="bull_low_vol",
        new_regime="bear_high_vol",
        exposure_scale=0.25
    )
    
    # Send drawdown alert
    alerter.send_drawdown_alert(
        current_drawdown=0.18,
        threshold=0.15,
        is_throttled=True
    )
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from loguru import logger

# Optional imports for HTTP requests (Slack)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertManager:
    """
    Manages alerts via email and/or Slack.
    
    Reads configuration from environment variables.
    Gracefully handles missing configuration (logs warnings but doesn't fail).
    """
    
    def __init__(
        self,
        email_enabled: bool = True,
        slack_enabled: bool = True,
        dry_run: bool = False
    ):
        """
        Initialize alert manager.
        
        Args:
            email_enabled: Whether to attempt email alerts
            slack_enabled: Whether to attempt Slack alerts
            dry_run: If True, log alerts but don't actually send them
        """
        self.dry_run = dry_run
        self.email_enabled = email_enabled
        self.slack_enabled = slack_enabled
        
        # Email configuration
        self.smtp_host = os.environ.get("ALERT_EMAIL_SMTP_HOST")
        self.smtp_port = int(os.environ.get("ALERT_EMAIL_SMTP_PORT", "587"))
        self.email_username = os.environ.get("ALERT_EMAIL_USERNAME")
        self.email_password = os.environ.get("ALERT_EMAIL_PASSWORD")
        self.email_from = os.environ.get("ALERT_EMAIL_FROM")
        self.email_to = os.environ.get("ALERT_EMAIL_TO", "").split(",")
        self.email_to = [e.strip() for e in self.email_to if e.strip()]
        
        # Slack configuration
        self.slack_webhook_url = os.environ.get("ALERT_SLACK_WEBHOOK_URL")
        
        # Check configuration
        self._email_configured = all([
            self.smtp_host,
            self.email_username,
            self.email_password,
            self.email_from,
            len(self.email_to) > 0
        ])
        
        self._slack_configured = bool(self.slack_webhook_url) and REQUESTS_AVAILABLE
        
        if email_enabled and not self._email_configured:
            logger.warning("Email alerting enabled but not configured (missing env vars)")
        
        if slack_enabled and not self._slack_configured:
            if not REQUESTS_AVAILABLE:
                logger.warning("Slack alerting requires 'requests' package")
            else:
                logger.warning("Slack alerting enabled but not configured (missing ALERT_SLACK_WEBHOOK_URL)")
    
    def _send_email(
        self,
        subject: str,
        body: str,
        level: AlertLevel = AlertLevel.INFO
    ) -> bool:
        """Send email alert."""
        if not self.email_enabled or not self._email_configured:
            return False
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would send email: {subject}")
            return True
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = ", ".join(self.email_to)
            msg['Subject'] = f"[{level.value.upper()}] {subject}"
            
            # Add level indicator to body
            full_body = f"Alert Level: {level.value.upper()}\n"
            full_body += f"Timestamp: {datetime.now().isoformat()}\n"
            full_body += "-" * 50 + "\n\n"
            full_body += body
            
            msg.attach(MIMEText(full_body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_slack(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        fields: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send Slack alert via webhook."""
        if not self.slack_enabled or not self._slack_configured:
            return False
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would send Slack: {message}")
            return True
        
        try:
            # Color based on level
            color_map = {
                AlertLevel.INFO: "#36a64f",      # Green
                AlertLevel.WARNING: "#ffcc00",   # Yellow
                AlertLevel.ERROR: "#ff6600",     # Orange
                AlertLevel.CRITICAL: "#ff0000",  # Red
            }
            
            # Build Slack attachment
            attachment = {
                "color": color_map.get(level, "#36a64f"),
                "title": f"Trading Alert - {level.value.upper()}",
                "text": message,
                "ts": int(datetime.now().timestamp()),
            }
            
            if fields:
                attachment["fields"] = [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in fields.items()
                ]
            
            payload = {
                "attachments": [attachment]
            }
            
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def send_alert(
        self,
        subject: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        fields: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send alert via all configured channels.
        
        Args:
            subject: Alert subject (used for email)
            message: Alert message body
            level: Alert severity level
            fields: Optional key-value pairs for structured data
        
        Returns:
            True if at least one channel succeeded
        """
        success = False
        
        # Build full message for email
        email_body = message
        if fields:
            email_body += "\n\nDetails:\n"
            for k, v in fields.items():
                email_body += f"  {k}: {v}\n"
        
        # Send via all channels
        if self._send_email(subject, email_body, level):
            success = True
        
        if self._send_slack(message, level, fields):
            success = True
        
        return success
    
    def send_regime_change_alert(
        self,
        old_regime: str,
        new_regime: str,
        exposure_scale: float,
        additional_info: Optional[Dict] = None
    ) -> bool:
        """
        Send alert for regime change.
        
        Args:
            old_regime: Previous regime descriptor
            new_regime: New regime descriptor
            exposure_scale: New exposure scale factor
            additional_info: Optional additional context
        """
        # Determine severity based on new regime
        if new_regime == "bear_high_vol":
            level = AlertLevel.CRITICAL
        elif new_regime in ("bear_low_vol", "bull_high_vol"):
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO
        
        subject = f"Regime Change: {old_regime} → {new_regime}"
        message = (
            f"Market regime has changed from {old_regime} to {new_regime}.\n"
            f"Exposure scale adjusted to {exposure_scale:.0%}."
        )
        
        fields = {
            "Old Regime": old_regime,
            "New Regime": new_regime,
            "Exposure Scale": f"{exposure_scale:.0%}",
        }
        
        if additional_info:
            fields.update({k: str(v) for k, v in additional_info.items()})
        
        return self.send_alert(subject, message, level, fields)
    
    def send_drawdown_alert(
        self,
        current_drawdown: float,
        threshold: float,
        is_throttled: bool,
        scale_factor: Optional[float] = None
    ) -> bool:
        """
        Send alert for drawdown threshold breach.
        
        Args:
            current_drawdown: Current drawdown as decimal (e.g., 0.15 for 15%)
            threshold: Threshold that was crossed
            is_throttled: Whether throttling is now active
            scale_factor: Current position scale factor if throttled
        """
        level = AlertLevel.ERROR if is_throttled else AlertLevel.WARNING
        
        subject = f"Drawdown Alert: {current_drawdown:.1%}"
        
        if is_throttled:
            message = (
                f"Drawdown of {current_drawdown:.1%} exceeds threshold of {threshold:.1%}.\n"
                f"Position sizing has been THROTTLED to {scale_factor:.0%} of normal."
            )
        else:
            message = (
                f"Drawdown of {current_drawdown:.1%} is approaching threshold of {threshold:.1%}.\n"
                "Monitoring closely."
            )
        
        fields = {
            "Current Drawdown": f"{current_drawdown:.1%}",
            "Threshold": f"{threshold:.1%}",
            "Throttled": "Yes" if is_throttled else "No",
        }
        
        if scale_factor is not None:
            fields["Scale Factor"] = f"{scale_factor:.0%}"
        
        return self.send_alert(subject, message, level, fields)
    
    def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Send alert for system errors.
        
        Args:
            error_type: Type of error (e.g., "Data Fetch", "Model Prediction")
            error_message: Error details
            context: Optional context information
        """
        subject = f"System Error: {error_type}"
        message = f"An error occurred in the trading system:\n\n{error_message}"
        
        fields = {"Error Type": error_type}
        if context:
            fields.update({k: str(v) for k, v in context.items()})
        
        return self.send_alert(subject, message, AlertLevel.ERROR, fields)
    
    def send_daily_summary(
        self,
        trading_date: str,
        regime: str,
        exposure_scale: float,
        account_value: float,
        positions: int,
        orders: int,
        pnl: Optional[float] = None,
        additional_metrics: Optional[Dict] = None
    ) -> bool:
        """
        Send daily trading summary.
        
        Args:
            trading_date: Date string
            regime: Current regime descriptor
            exposure_scale: Current exposure scale
            account_value: Total account value
            positions: Number of positions
            orders: Number of orders submitted
            pnl: Day's P&L if available
            additional_metrics: Optional additional metrics
        """
        subject = f"Daily Summary - {trading_date}"
        message = (
            f"Trading completed for {trading_date}.\n\n"
            f"Regime: {regime}\n"
            f"Exposure: {exposure_scale:.0%}\n"
            f"Account Value: ${account_value:,.2f}\n"
            f"Positions: {positions}\n"
            f"Orders: {orders}"
        )
        
        if pnl is not None:
            message += f"\nDay P&L: ${pnl:,.2f}"
        
        fields = {
            "Date": trading_date,
            "Regime": regime,
            "Exposure": f"{exposure_scale:.0%}",
            "Account Value": f"${account_value:,.2f}",
            "Positions": str(positions),
            "Orders": str(orders),
        }
        
        if pnl is not None:
            fields["Day P&L"] = f"${pnl:,.2f}"
        
        if additional_metrics:
            fields.update({k: str(v) for k, v in additional_metrics.items()})
        
        return self.send_alert(subject, message, AlertLevel.INFO, fields)


def send_test_alert():
    """Send a test alert to verify configuration."""
    alerter = AlertManager()
    
    success = alerter.send_alert(
        subject="Test Alert",
        message="This is a test alert from the py-finance trading system.",
        level=AlertLevel.INFO,
        fields={
            "Test": "True",
            "Timestamp": datetime.now().isoformat(),
        }
    )
    
    if success:
        print("Test alert sent successfully!")
    else:
        print("Failed to send test alert. Check configuration.")
    
    return success


if __name__ == "__main__":
    send_test_alert()

