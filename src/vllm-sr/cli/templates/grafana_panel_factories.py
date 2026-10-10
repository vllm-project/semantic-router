"""Grafana panel JSON factories for Prometheus-backed dashboards."""


def metric(name, labels=""):
    """Bind every router query to the selected scrape instances."""
    selectors = 'job="semantic-router",instance=~"${router_instance:regex}"'
    return f"{name}{{{selectors}{',' + labels if labels else ''}}}"


def rate_sum(name, group="", labels=""):
    grouped = f" by ({group})" if group else ""
    return f"sum(rate({metric(name, labels)}[$__rate_interval])){grouped}"


def histogram_mean(name, group):
    """Means retain long observations; empty populations stay absent."""
    numerator = rate_sum(name + "_sum", group)
    denominator = rate_sum(name + "_count", group)
    return f"{numerator} / ({denominator} > 0)"


def create_stat_panel(title, expr, unit="short", x=0, y=0, w=6, h=6, panel_id=1):
    """Create a stat panel"""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "unit": unit,
            }
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": panel_id,
        "options": {
            "colorMode": "none",
            "graphMode": "area",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
        },
        "pluginVersion": "11.5.1",
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                "expr": expr,
                "refId": "A",
            }
        ],
        "title": title,
        "type": "stat",
    }


def create_timeseries_panel(
    title, targets, x=0, y=0, w=12, h=8, panel_id=1, unit="short"
):
    """Create a time series panel"""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "hideFrom": {"tooltip": False, "viz": False, "legend": False},
                    "insertNulls": False,
                    "lineInterpolation": "linear",
                    "lineWidth": 1,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "never",
                    "spanNulls": False,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "off"},
                },
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "unit": unit,
            }
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": panel_id,
        "options": {
            "legend": {
                "calcs": [],
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi", "sort": "none"},
        },
        "pluginVersion": "11.5.1",
        "targets": targets,
        "title": title,
        "type": "timeseries",
    }


def create_row_panel(title, y=0, panel_id=100):
    """Create a row panel"""
    return {
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "id": panel_id,
        "panels": [],
        "title": title,
        "type": "row",
    }


def create_target(expr, legend="", ref_id="A"):
    """Create a query target"""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "expr": expr,
        "legendFormat": legend,
        "refId": ref_id,
    }
