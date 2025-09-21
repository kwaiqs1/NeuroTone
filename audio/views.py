import os
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.files.storage import default_storage
from .forms import UploadForm
from .models import AudioUpload
from .pipeline import get_pipeline, PipelineConfig




def index(request):
    form = UploadForm()
    return render(request, 'audio/index.html', {'form': form})




def upload_and_process(request):
    if request.method != 'POST':
        return redirect('audio:index')


    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, 'audio/index.html', {'form': form})


    f = request.FILES['audio']
    # Save original
    item = AudioUpload.objects.create(original_file=f)


# Prepare output path
    in_path = item.original_file.path
    out_rel = os.path.join('processed', f'processed_{item.pk}.wav')
    out_abs = os.path.join(settings.MEDIA_ROOT, out_rel)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)




    ts = form.cleaned_data.get('trigger_sensitivity')
    if ts in (None, ''):
        ts = 0.15  # дефолт агрессивности

    vs = form.cleaned_data.get('vacuum_strength')
    if vs in (None, ''):
        vs = 0.8  # дефолт силы "вакуума"


    # Run pipeline
    cfg = PipelineConfig(
        base_denoise=form.cleaned_data.get('apply_base_denoise', True),
        suppress_triggers=form.cleaned_data.get('suppress_triggers', True),
        trigger_sensitivity=float(ts),
        preserve_speech=form.cleaned_data.get('preserve_speech', True),
        vacuum_mode=form.cleaned_data.get('vacuum_mode', True),
        vacuum_strength=float(vs),
    )


    try:
        pipeline = get_pipeline()
        info = pipeline.process_file(in_path, out_abs, config=cfg)
        # Attach processed file to model
        with open(out_abs, 'rb') as fh:
            item.processed_file.save(os.path.basename(out_abs), fh, save=True)
        item.notes = (
            'Trigger scores (0..1):\n' +
            '\n'.join([f" - {k}: {v:.3f}" for k, v in info.get('triggers', {}).items()])
        )
        item.save()
    except Exception as e:
        item.notes = f'Processing error: {e}'
        item.save()


    return redirect('audio:detail', pk=item.pk)




def detail(request, pk):
    obj = get_object_or_404(AudioUpload, pk=pk)
    return render(request, 'audio/detail.html', {'object': obj, 'notes': obj.notes})