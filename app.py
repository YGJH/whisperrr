from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort, jsonify
from flask_cors import CORS
import threading
import subprocess
import uuid
import os
import sys
import time

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(APP_ROOT, 'jobs')
if not os.path.exists(JOBS_DIR):
    os.makedirs(JOBS_DIR)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
JOBS = {}  # jobid -> {status, log_path, summary_path}


def run_job(jobid, url, model, system_prompt, copy_flag):
    """
    Run transcription job in background thread.
    All errors are caught and logged to ensure user always sees what went wrong.
    """
    jobdir = os.path.join(JOBS_DIR, jobid)
    os.makedirs(jobdir, exist_ok=True)
    log_path = os.path.join(jobdir, 'run.log')
    summary_dest = os.path.join(jobdir, 'summary.md')

    JOBS[jobid] = {'status': 'running', 'log': log_path, 'summary': None}

    try:
        cmd = ['uv', 'run', os.path.join(APP_ROOT, 'main.py'), '--video', url]
        if model:
            cmd += ['--model', model]
        if system_prompt:
            cmd += ['--system-prompt', system_prompt]
        if copy_flag:
            cmd += ['--copy']
        # Always request summary so main.py will generate summary.md
        if '--summary' not in cmd:
            cmd += ['--summary']

        # Write command for debugging
        with open(log_path, 'w', encoding='utf-8') as lf:
            lf.write('=' * 80 + '\n')
            lf.write('WHISPER TRANSCRIPTION JOB\n')
            lf.write('=' * 80 + '\n')
            lf.write(f'Job ID: {jobid}\n')
            lf.write(f'URL: {url}\n')
            lf.write(f'Model: {model}\n')
            lf.write(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            lf.write('-' * 80 + '\n')
            lf.write('Command: ' + ' '.join(cmd) + '\n')
            lf.write('=' * 80 + '\n\n')
            lf.flush()

        # Stream process output to log file
        env = os.environ.copy()
        # pass job dir to the subprocess so it can write progress.json
        env['JOB_DIR'] = jobdir
        
        try:
            proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=env,
                cwd=APP_ROOT  # Ensure we run from the correct directory
            )
            
            with open(log_path, 'a', encoding='utf-8') as lf:
                if proc.stdout:
                    for line in proc.stdout:
                        lf.write(line)
                        lf.flush()
            
            ret = proc.wait()
            
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write('\n' + '=' * 80 + '\n')
                lf.write(f'Process exited with code {ret}\n')
                lf.write('=' * 80 + '\n')
            
            # Check if process failed
            if ret != 0:
                JOBS[jobid]['status'] = 'failed'
                with open(log_path, 'a', encoding='utf-8') as lf:
                    lf.write(f'\n❌ ERROR: Process exited with non-zero code {ret}\n')
                return
            
        except FileNotFoundError as e:
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(f'\n❌ ERROR: Command not found\n')
                lf.write(f'Details: {str(e)}\n')
                lf.write(f'Make sure "uv" is installed and in PATH\n')
            JOBS[jobid]['status'] = 'failed'
            return
        
        except Exception as e:
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(f'\n❌ ERROR: Failed to execute command\n')
                lf.write(f'Details: {str(e)}\n')
                import traceback
                lf.write(f'Traceback:\n{traceback.format_exc()}\n')
            JOBS[jobid]['status'] = 'failed'
            return

        # If main.py produced summary.md in repo root, copy it to job dir
        generated = os.path.join(APP_ROOT, 'summary.md')
        if os.path.exists(generated):
            try:
                # overwrite if exists
                if os.path.exists(summary_dest):
                    os.remove(summary_dest)
                os.replace(generated, summary_dest)
                JOBS[jobid]['summary'] = summary_dest
                JOBS[jobid]['status'] = 'finished'
                with open(log_path, 'a', encoding='utf-8') as lf:
                    lf.write('\n✅ SUCCESS: Summary generated successfully\n')
            except Exception as e:
                with open(log_path, 'a', encoding='utf-8') as lf:
                    lf.write(f'\n⚠️  WARNING: Failed to move summary: {e}\n')
                JOBS[jobid]['status'] = 'finished'
        else:
            JOBS[jobid]['status'] = 'finished'
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write('\n⚠️  WARNING: No summary.md file was generated\n')
                
    except Exception as e:
        # Catch any unexpected errors in the job runner itself
        try:
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(f'\n❌ CRITICAL ERROR in job runner\n')
                lf.write(f'Details: {str(e)}\n')
                import traceback
                lf.write(f'Traceback:\n{traceback.format_exc()}\n')
        except:
            # If we can't even write to log file, print to stderr
            print(f'CRITICAL: Job {jobid} failed and could not write to log: {e}', file=sys.stderr)
        
        JOBS[jobid]['status'] = 'failed'


@app.route('/')
def index():
    """Redirect to React frontend running on port 3000."""
    # In production, you would serve the built React app here
    # For now, redirect to the Next.js dev server
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Whisper Transcription</title>
        <meta http-equiv="refresh" content="0; url=http://localhost:3000" />
    </head>
    <body>
        <p>Redirecting to React app... If not redirected, <a href="http://localhost:3000">click here</a>.</p>
    </body>
    </html>
    '''

@app.route('/html')
def html_index():
    """Original HTML template version (for backward compatibility)."""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_api():
    """API endpoint for starting transcription jobs (for React frontend)."""
    data = request.get_json()
    
    url = data.get('url')
    model = data.get('model')
    system_prompt = data.get('systemPrompt')
    copy_flag = data.get('copyToPCloud', True)

    if not url:
        return jsonify({'error': 'Missing URL'}), 400

    jobid = uuid.uuid4().hex[:12]
    t = threading.Thread(target=run_job, args=(jobid, url, model, system_prompt, copy_flag), daemon=True)
    t.start()

    return jsonify({'jobId': jobid, 'status': 'started'}), 200


@app.route('/start', methods=['POST'])
def start():
    """HTML form endpoint (for backward compatibility)."""
    url = request.form.get('url')
    model = request.form.get('model')
    system_prompt = request.form.get('system_prompt')
    copy_flag = True if request.form.get('copy') == 'on' else False

    if not url:
        return 'Missing URL', 400

    jobid = uuid.uuid4().hex[:12]
    t = threading.Thread(target=run_job, args=(jobid, url, model, system_prompt, copy_flag), daemon=True)
    t.start()

    return redirect(url_for('status', jobid=jobid))


@app.route('/status/<jobid>')
def status(jobid):
    """Display job status page with logs and progress."""
    if jobid not in JOBS:
        # It may exist as dir but not started here
        jobdir = os.path.join(JOBS_DIR, jobid)
        if os.path.exists(jobdir):
            log_file = os.path.join(jobdir, 'run.log')
            # Try to infer status from log file
            job_status = 'unknown'
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '❌' in content or 'ERROR' in content:
                            job_status = 'failed'
                        elif '✅ SUCCESS' in content:
                            job_status = 'finished'
                        elif 'Process exited with code' in content:
                            if 'code 0' in content:
                                job_status = 'finished'
                            else:
                                job_status = 'failed'
                except:
                    pass
            
            JOBS[jobid] = {'status': job_status, 'log': log_file, 'summary': None}
        else:
            abort(404)

    info = JOBS[jobid]
    log_content = ''
    progress = None
    
    # Read log file
    if os.path.exists(info['log']):
        try:
            with open(info['log'], 'r', encoding='utf-8') as lf:
                log_content = lf.read()
        except Exception as e:
            log_content = f'Unable to read log file: {e}'
    else:
        log_content = 'Log file not yet created. Job may be starting...'

    # Read progress.json if present
    jobdir = os.path.join(JOBS_DIR, jobid)
    progress_file = os.path.join(jobdir, 'progress.json')
    if os.path.exists(progress_file):
        try:
            import json
            with open(progress_file, 'r', encoding='utf-8') as pf:
                progress = json.load(pf)
        except Exception:
            progress = None

    # Check for summary file
    summary_available = info.get('summary') and os.path.exists(info.get('summary'))
    
    return render_template(
        'status.html', 
        jobid=jobid, 
        status=info['status'], 
        log=log_content, 
        summary_available=summary_available, 
        progress=progress
    )


@app.route('/api/download/<jobid>')
@app.route('/download/<jobid>')
def download(jobid):
    info = JOBS.get(jobid)
    if not info:
        # Try to find in job directory
        jobdir = os.path.join(JOBS_DIR, jobid)
        summary_dest = os.path.join(jobdir, 'summary.md')
        if os.path.exists(summary_dest):
            return send_from_directory(os.path.dirname(summary_dest), os.path.basename(summary_dest), as_attachment=True)
        abort(404)
    
    summary = info.get('summary')
    if summary and os.path.exists(summary):
        return send_from_directory(os.path.dirname(summary), os.path.basename(summary), as_attachment=True)
    # fallback: if repo summary.md exists and job finished, serve that
    repo_summary = os.path.join(APP_ROOT, 'summary.md')
    if os.path.exists(repo_summary):
        return send_from_directory(APP_ROOT, 'summary.md', as_attachment=True)
    abort(404)


@app.route('/api/job/<jobid>')
def job_status_api(jobid):
    """Get full job status including progress and logs."""
    if jobid not in JOBS:
        jobdir = os.path.join(JOBS_DIR, jobid)
        if os.path.exists(jobdir):
            log_file = os.path.join(jobdir, 'run.log')
            job_status = 'unknown'
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '❌' in content or 'ERROR' in content:
                            job_status = 'failed'
                        elif '✅ SUCCESS' in content:
                            job_status = 'finished'
                        elif 'Process exited with code' in content:
                            if 'code 0' in content:
                                job_status = 'finished'
                            else:
                                job_status = 'failed'
                except:
                    pass
            JOBS[jobid] = {'status': job_status, 'log': log_file, 'summary': None}
        else:
            return jsonify({'error': 'Job not found'}), 404

    info = JOBS[jobid]
    log_content = ''
    progress = None
    
    # Read log file
    if os.path.exists(info['log']):
        try:
            with open(info['log'], 'r', encoding='utf-8') as lf:
                log_content = lf.read()
        except Exception as e:
            log_content = f'Unable to read log file: {e}'

    # Read progress.json if present
    jobdir = os.path.join(JOBS_DIR, jobid)
    progress_file = os.path.join(jobdir, 'progress.json')
    if os.path.exists(progress_file):
        try:
            import json
            with open(progress_file, 'r', encoding='utf-8') as pf:
                progress = json.load(pf)
        except Exception:
            progress = None

    # Check for summary file
    summary_available = info.get('summary') and os.path.exists(info.get('summary'))
    
    return jsonify({
        'id': jobid,
        'status': info['status'],
        'log': log_content,
        'progress': progress,
        'summaryAvailable': summary_available
    })


@app.route('/progress/<jobid>')
def progress_api(jobid):
    # Return JSON progress for the job if present
    jobdir = os.path.join(JOBS_DIR, jobid)
    progress_file = os.path.join(jobdir, 'progress.json')
    if os.path.exists(progress_file):
        try:
            import json
            with open(progress_file, 'r', encoding='utf-8') as pf:
                data = json.load(pf)
            return jsonify({'ok': True, 'progress': data})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    return jsonify({'ok': False, 'error': 'no progress file'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
