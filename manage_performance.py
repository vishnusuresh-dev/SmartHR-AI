#!/usr/bin/env python3
"""
Performance Metrics Management Script
======================================
Use this script to manage employee performance metrics from the command line.

Usage:
    python manage_performance.py list                              # List all employees
    python manage_performance.py view EMP001                       # View performance for employee
    python manage_performance.py update EMP001                     # Update performance interactively
    python manage_performance.py set EMP001 --attendance 95        # Set specific metric
    python manage_performance.py bulk-update employees.csv         # Bulk update from CSV
    python manage_performance.py reset EMP001                      # Reset to defaults
    python manage_performance.py export report.csv                 # Export all metrics
"""

import sys
import os
import argparse
import csv
from datetime import datetime
from tabulate import tabulate

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Employee, PerformanceMetric, get_or_create_performance_metric


def list_employees():
    """List all employees with their current performance scores"""
    with app.app_context():
        employees = Employee.query.all()
        
        if not employees:
            print("No employees found in database.")
            return
        
        data = []
        for emp in employees:
            current_month = datetime.utcnow().strftime("%Y-%m")
            metric = PerformanceMetric.query.filter_by(
                employee_id=emp.employee_id,
                month=current_month
            ).first()
            
            score = metric.calculate_overall_score() if metric else "N/A"
            
            data.append([
                emp.employee_id,
                emp.full_name,
                emp.department or "N/A",
                emp.job_title or "N/A",
                score
            ])
        
        headers = ["Employee ID", "Name", "Department", "Job Title", "Performance Score"]
        print("\n" + tabulate(data, headers=headers, tablefmt="grid"))
        print(f"\nTotal Employees: {len(employees)}\n")


def view_employee_performance(employee_id, month=None):
    """View detailed performance metrics for an employee"""
    with app.app_context():
        employee = Employee.query.filter_by(employee_id=employee_id).first()
        
        if not employee:
            print(f"❌ Employee {employee_id} not found!")
            return
        
        if month is None:
            month = datetime.utcnow().strftime("%Y-%m")
        
        metric = PerformanceMetric.query.filter_by(
            employee_id=employee_id,
            month=month
        ).first()
        
        if not metric:
            print(f"⚠️  No performance data found for {employee_id} in {month}")
            print("Creating default metrics...")
            metric = get_or_create_performance_metric(employee_id, month)
        
        print(f"\n{'='*60}")
        print(f"Performance Report: {employee.full_name} ({employee_id})")
        print(f"Month: {month}")
        print(f"{'='*60}\n")
        
        print(f"Overall Score: {metric.calculate_overall_score()}/100\n")
        
        # Attendance
        print("📅 ATTENDANCE")
        print(f"  Score: {metric.attendance_score}/100")
        print(f"  Days Present: {metric.days_present}/{metric.days_total}")
        print(f"  Late Arrivals: {metric.late_arrivals}\n")
        
        # Task Completion
        print("✅ TASK COMPLETION")
        print(f"  Score: {metric.task_completion_score}/100")
        print(f"  Tasks: {metric.tasks_completed}/{metric.tasks_assigned}")
        print(f"  On-Time: {metric.on_time_completion}%\n")
        
        # Quality
        print("⭐ QUALITY")
        print(f"  Score: {metric.quality_score}/100")
        print(f"  Bug Rate: {metric.bug_rate}%")
        print(f"  Review Rating: {metric.review_rating}/5.0")
        print(f"  Rework Required: {metric.rework_required}%\n")
        
        # Punctuality
        print("⏰ PUNCTUALITY")
        print(f"  Score: {metric.punctuality_score}/100")
        print(f"  Meeting Attendance: {metric.meeting_attendance}%")
        print(f"  Deadline Adherence: {metric.deadline_adherence}%\n")
        
        # Collaboration
        print("🤝 COLLABORATION")
        print(f"  Score: {metric.collaboration_score}/100")
        print(f"  Peer Reviews: {metric.peer_reviews}")
        print(f"  Team Contributions: {metric.team_contributions}")
        print(f"  Communication Rating: {metric.communication_rating}/5.0\n")
        
        # Productivity
        print("📊 PRODUCTIVITY")
        print(f"  Score: {metric.productivity_score}/100")
        print(f"  Lines of Code: {metric.lines_of_code}")
        print(f"  Commits: {metric.commits}")
        print(f"  Story Points: {metric.story_points}\n")
        
        if metric.notes:
            print(f"📝 Notes: {metric.notes}\n")
        
        print(f"Last Updated: {metric.last_updated.strftime('%Y-%m-%d %H:%M:%S')}\n")


def update_employee_interactive(employee_id):
    """Interactively update employee performance metrics"""
    with app.app_context():
        employee = Employee.query.filter_by(employee_id=employee_id).first()
        
        if not employee:
            print(f"❌ Employee {employee_id} not found!")
            return
        
        month = datetime.utcnow().strftime("%Y-%m")
        metric = get_or_create_performance_metric(employee_id, month)
        
        print(f"\n{'='*60}")
        print(f"Update Performance: {employee.full_name} ({employee_id})")
        print(f"Month: {month}")
        print(f"{'='*60}\n")
        print("Press Enter to keep current value\n")
        
        # Attendance
        print("📅 ATTENDANCE")
        metric.attendance_score = float(input(f"  Attendance Score [{metric.attendance_score}]: ") or metric.attendance_score)
        metric.days_present = int(input(f"  Days Present [{metric.days_present}]: ") or metric.days_present)
        metric.days_total = int(input(f"  Days Total [{metric.days_total}]: ") or metric.days_total)
        metric.late_arrivals = int(input(f"  Late Arrivals [{metric.late_arrivals}]: ") or metric.late_arrivals)
        
        # Task Completion
        print("\n✅ TASK COMPLETION")
        metric.task_completion_score = float(input(f"  Task Completion Score [{metric.task_completion_score}]: ") or metric.task_completion_score)
        metric.tasks_completed = int(input(f"  Tasks Completed [{metric.tasks_completed}]: ") or metric.tasks_completed)
        metric.tasks_assigned = int(input(f"  Tasks Assigned [{metric.tasks_assigned}]: ") or metric.tasks_assigned)
        metric.on_time_completion = float(input(f"  On-Time Completion % [{metric.on_time_completion}]: ") or metric.on_time_completion)
        
        # Quality
        print("\n⭐ QUALITY")
        metric.quality_score = float(input(f"  Quality Score [{metric.quality_score}]: ") or metric.quality_score)
        metric.bug_rate = float(input(f"  Bug Rate % [{metric.bug_rate}]: ") or metric.bug_rate)
        metric.review_rating = float(input(f"  Review Rating (0-5) [{metric.review_rating}]: ") or metric.review_rating)
        metric.rework_required = float(input(f"  Rework Required % [{metric.rework_required}]: ") or metric.rework_required)
        
        # Punctuality
        print("\n⏰ PUNCTUALITY")
        metric.punctuality_score = float(input(f"  Punctuality Score [{metric.punctuality_score}]: ") or metric.punctuality_score)
        metric.meeting_attendance = float(input(f"  Meeting Attendance % [{metric.meeting_attendance}]: ") or metric.meeting_attendance)
        metric.deadline_adherence = float(input(f"  Deadline Adherence % [{metric.deadline_adherence}]: ") or metric.deadline_adherence)
        
        # Collaboration
        print("\n🤝 COLLABORATION")
        metric.collaboration_score = float(input(f"  Collaboration Score [{metric.collaboration_score}]: ") or metric.collaboration_score)
        metric.peer_reviews = int(input(f"  Peer Reviews [{metric.peer_reviews}]: ") or metric.peer_reviews)
        metric.team_contributions = int(input(f"  Team Contributions [{metric.team_contributions}]: ") or metric.team_contributions)
        metric.communication_rating = float(input(f"  Communication Rating (0-5) [{metric.communication_rating}]: ") or metric.communication_rating)
        
        # Productivity
        print("\n📊 PRODUCTIVITY")
        metric.productivity_score = float(input(f"  Productivity Score [{metric.productivity_score}]: ") or metric.productivity_score)
        metric.lines_of_code = int(input(f"  Lines of Code [{metric.lines_of_code}]: ") or metric.lines_of_code)
        metric.commits = int(input(f"  Commits [{metric.commits}]: ") or metric.commits)
        metric.story_points = int(input(f"  Story Points [{metric.story_points}]: ") or metric.story_points)
        
        # Notes
        print()
        notes = input(f"  Notes [{metric.notes or 'None'}]: ")
        if notes:
            metric.notes = notes
        
        # Save
        metric.last_updated = datetime.utcnow()
        employee.performance_score = metric.calculate_overall_score()
        db.session.commit()
        
        print(f"\n✅ Performance metrics updated successfully!")
        print(f"Overall Score: {employee.performance_score}/100\n")


def set_specific_metric(employee_id, metric_name, value):
    """Set a specific metric value"""
    with app.app_context():
        employee = Employee.query.filter_by(employee_id=employee_id).first()
        
        if not employee:
            print(f"❌ Employee {employee_id} not found!")
            return
        
        month = datetime.utcnow().strftime("%Y-%m")
        metric = get_or_create_performance_metric(employee_id, month)
        
        # Map of allowed metric names
        metric_map = {
            'attendance': 'attendance_score',
            'task_completion': 'task_completion_score',
            'quality': 'quality_score',
            'punctuality': 'punctuality_score',
            'collaboration': 'collaboration_score',
            'productivity': 'productivity_score',
            'days_present': 'days_present',
            'days_total': 'days_total',
            'late_arrivals': 'late_arrivals',
            'tasks_completed': 'tasks_completed',
            'tasks_assigned': 'tasks_assigned',
            'bug_rate': 'bug_rate',
            'commits': 'commits',
            'lines_of_code': 'lines_of_code'
        }
        
        if metric_name not in metric_map:
            print(f"❌ Unknown metric: {metric_name}")
            print(f"Available metrics: {', '.join(metric_map.keys())}")
            return
        
        attr_name = metric_map[metric_name]
        
        try:
            # Convert to appropriate type
            if attr_name in ['days_present', 'days_total', 'late_arrivals', 'tasks_completed', 
                             'tasks_assigned', 'peer_reviews', 'team_contributions', 
                             'lines_of_code', 'commits', 'story_points']:
                value = int(value)
            else:
                value = float(value)
            
            setattr(metric, attr_name, value)
            metric.last_updated = datetime.utcnow()
            employee.performance_score = metric.calculate_overall_score()
            db.session.commit()
            
            print(f"✅ Updated {employee.full_name}'s {metric_name} to {value}")
            print(f"Overall Score: {employee.performance_score}/100")
            
        except ValueError:
            print(f"❌ Invalid value: {value}")


def bulk_update_from_csv(csv_file):
    """Bulk update performance metrics from CSV file"""
    with app.app_context():
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            return
        
        updated = 0
        errors = 0
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    employee_id = row.get('employee_id')
                    employee = Employee.query.filter_by(employee_id=employee_id).first()
                    
                    if not employee:
                        print(f"⚠️  Employee {employee_id} not found, skipping...")
                        errors += 1
                        continue
                    
                    month = row.get('month', datetime.utcnow().strftime("%Y-%m"))
                    metric = get_or_create_performance_metric(employee_id, month)
                    
                    # Update all provided fields
                    if 'attendance_score' in row:
                        metric.attendance_score = float(row['attendance_score'])
                    if 'task_completion_score' in row:
                        metric.task_completion_score = float(row['task_completion_score'])
                    if 'quality_score' in row:
                        metric.quality_score = float(row['quality_score'])
                    if 'punctuality_score' in row:
                        metric.punctuality_score = float(row['punctuality_score'])
                    if 'collaboration_score' in row:
                        metric.collaboration_score = float(row['collaboration_score'])
                    if 'productivity_score' in row:
                        metric.productivity_score = float(row['productivity_score'])
                    
                    metric.last_updated = datetime.utcnow()
                    employee.performance_score = metric.calculate_overall_score()
                    
                    updated += 1
                    print(f"✅ Updated {employee.full_name} ({employee_id})")
                    
                except Exception as e:
                    print(f"❌ Error updating {employee_id}: {e}")
                    errors += 1
        
        db.session.commit()
        print(f"\n{'='*60}")
        print(f"Bulk Update Complete")
        print(f"Updated: {updated} | Errors: {errors}")
        print(f"{'='*60}\n")


def export_to_csv(output_file):
    """Export all performance metrics to CSV"""
    with app.app_context():
        current_month = datetime.utcnow().strftime("%Y-%m")
        employees = Employee.query.all()
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = [
                'employee_id', 'full_name', 'department', 'month',
                'overall_score', 'attendance_score', 'task_completion_score',
                'quality_score', 'punctuality_score', 'collaboration_score', 
                'productivity_score', 'days_present', 'days_total', 
                'tasks_completed', 'tasks_assigned', 'notes', 'last_updated'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for emp in employees:
                metric = PerformanceMetric.query.filter_by(
                    employee_id=emp.employee_id,
                    month=current_month
                ).first()
                
                if not metric:
                    metric = get_or_create_performance_metric(emp.employee_id, current_month)
                
                writer.writerow({
                    'employee_id': emp.employee_id,
                    'full_name': emp.full_name,
                    'department': emp.department or '',
                    'month': current_month,
                    'overall_score': metric.calculate_overall_score(),
                    'attendance_score': metric.attendance_score,
                    'task_completion_score': metric.task_completion_score,
                    'quality_score': metric.quality_score,
                    'punctuality_score': metric.punctuality_score,
                    'collaboration_score': metric.collaboration_score,
                    'productivity_score': metric.productivity_score,
                    'days_present': metric.days_present,
                    'days_total': metric.days_total,
                    'tasks_completed': metric.tasks_completed,
                    'tasks_assigned': metric.tasks_assigned,
                    'notes': metric.notes or '',
                    'last_updated': metric.last_updated.isoformat()
                })
        
        print(f"✅ Exported {len(employees)} employee records to {output_file}\n")


def reset_employee_metrics(employee_id):
    """Reset employee metrics to defaults"""
    with app.app_context():
        employee = Employee.query.filter_by(employee_id=employee_id).first()
        
        if not employee:
            print(f"❌ Employee {employee_id} not found!")
            return
        
        month = datetime.utcnow().strftime("%Y-%m")
        metric = PerformanceMetric.query.filter_by(
            employee_id=employee_id,
            month=month
        ).first()
        
        if metric:
            db.session.delete(metric)
        
        # Create new default metric
        new_metric = get_or_create_performance_metric(employee_id, month)
        employee.performance_score = new_metric.calculate_overall_score()
        db.session.commit()
        
        print(f"✅ Reset {employee.full_name}'s metrics to defaults")
        print(f"Overall Score: {employee.performance_score}/100\n")


def main():
    parser = argparse.ArgumentParser(description='Manage employee performance metrics')
    parser.add_argument('command', choices=['list', 'view', 'update', 'set', 'bulk-update', 'export', 'reset'],
                       help='Command to execute')
    parser.add_argument('employee_id', nargs='?', help='Employee ID')
    parser.add_argument('--metric', help='Metric name for set command')
    parser.add_argument('--value', help='Value for set command')
    parser.add_argument('--attendance', type=float, help='Attendance score')
    parser.add_argument('--task-completion', type=float, help='Task completion score')
    parser.add_argument('--quality', type=float, help='Quality score')
    parser.add_argument('--punctuality', type=float, help='Punctuality score')
    parser.add_argument('--collaboration', type=float, help='Collaboration score')
    parser.add_argument('--productivity', type=float, help='Productivity score')
    parser.add_argument('--file', help='CSV file for bulk operations')
    parser.add_argument('--month', help='Month in YYYY-MM format')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_employees()
    
    elif args.command == 'view':
        if not args.employee_id:
            print("❌ Employee ID required for view command")
            return
        view_employee_performance(args.employee_id, args.month)
    
    elif args.command == 'update':
        if not args.employee_id:
            print("❌ Employee ID required for update command")
            return
        update_employee_interactive(args.employee_id)
    
    elif args.command == 'set':
        if not args.employee_id:
            print("❌ Employee ID required for set command")
            return
        
        # Check for quick flags
        if args.attendance:
            set_specific_metric(args.employee_id, 'attendance', args.attendance)
        elif args.task_completion:
            set_specific_metric(args.employee_id, 'task_completion', args.task_completion)
        elif args.quality:
            set_specific_metric(args.employee_id, 'quality', args.quality)
        elif args.punctuality:
            set_specific_metric(args.employee_id, 'punctuality', args.punctuality)
        elif args.collaboration:
            set_specific_metric(args.employee_id, 'collaboration', args.collaboration)
        elif args.productivity:
            set_specific_metric(args.employee_id, 'productivity', args.productivity)
        elif args.metric and args.value:
            set_specific_metric(args.employee_id, args.metric, args.value)
        else:
            print("❌ Specify --metric and --value or use quick flags (--attendance, etc.)")
    
    elif args.command == 'bulk-update':
        csv_file = args.file or args.employee_id
        if not csv_file:
            print("❌ CSV file required for bulk-update")
            return
        bulk_update_from_csv(csv_file)
    
    elif args.command == 'export':
        output_file = args.file or args.employee_id or 'performance_export.csv'
        export_to_csv(output_file)
    
    elif args.command == 'reset':
        if not args.employee_id:
            print("❌ Employee ID required for reset command")
            return
        reset_employee_metrics(args.employee_id)


if __name__ == '__main__':
    main()